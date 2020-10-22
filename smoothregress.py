import math
import numpy as np
from lineardetectpack import generate_bfderivative_full, noise_signal_linear_regression
from jenksdetectpack import jenks_until
from peakdetectpure import peakdet
from scipy.optimize import curve_fit

def exponential_fit(x, a, b, c):
    return np.exp(a) * np.exp(b * x) + c

def smooth_data(data, window):
    averaged_current_case = []
    for i in range(window):
        averaged_current_case.append(data[i])
    for i in range(window, len(data[:-window])):
        windowed = data[ i-int(window/2) : i+1+int(window/2) ]
        averaged_current_case.append(  sum(windowed)/window)
    for i in range(len(data[:-window]), len(data)):
        averaged_current_case.append(data[i])
    return averaged_current_case

#from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth_scipy(data, window_size, window_type='flat'):
    data = np.array(data)
    window_type = window_type.lower()
    window_size = int(window_size)
    print("window_size")
    print(window_size)
    if data.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if data.size < window_size:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_size<3:
        return data
    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[data[window_size-1:0:-1],data,data[-2:-window_size-1:-1]]
    if window_type == 'flat': #moving average
        w=np.ones(window_size,'d')
    else:
        w=eval('np.'+window_type+'(window_size)')
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[math.ceil((window_size/2)-1):-math.ceil(window_size/2)]

def noise_definition(data):
    largs = noise_signal_linear_regression(data)
    if largs != False:
        ftoplot, ltoplot, xintersect, yintersect, xintersect_dot, yintersect_value = largs
        belowthresholdvalues = [e for e in sorted(data) if e < yintersect_value]
        return np.mean(belowthresholdvalues)
    else:
        #cutoff from avg first 25% lowest values
        sorted_data = sorted(data)
        sorted_data_25perc = sorted_data[: int(len(sorted_data) / 4)]
        return np.mean(sorted_data_25perc)

def _1gaussian(x, amp1,cen1,sigma1):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2)))

# def noise_detection(current_case, filter_noise_area=True, added_noise_dots=[], removed_noise_dots=[], cutoff_val=0.90):
def noise_detection(current_case, filter_noise_area=True, added_noise_dots=[], removed_noise_dots=[], cutoff_val=None):
    #TODO: Write up noise detection separately and pass args to peak_detection
    # print("current_case")
    # print(current_case)
    print("pre_cutoff_val")
    print(cutoff_val)
    if cutoff_val == None:
        # cutoff_val = np.median(current_case)
        # cutoff_val = float("{:.3f}".format(cutoff_val))
        n = int(len(current_case) * 0.3)
        vi = np.argsort(current_case)[-n:]
        cutoff_val = np.mean([current_case[a] for a in vi])
        cutoff_val = float("{:.3f}".format(cutoff_val))
    print("cutoff_val")
    print(cutoff_val)

    # gvf, case_classes = jenks_until(current_case, False, cutoff=cutoff_val)
    gvf = cutoff_val
    case_classes = []
    for e in current_case:
        if e < cutoff_val:
            case_classes.append(1)
        else:
            case_classes.append(2)
    
    # print("case_classes")
    # print(case_classes)

    #class > 1 are peak points
    non_noise_points = [(e, i) for i,e in enumerate(current_case) if case_classes[i] != 1]

    non_noise_points_values = [e[0] for e in non_noise_points]
    non_noise_points_indexes = [e[1] for e in non_noise_points]

    #class == 1 are noise points
    noise_points = None
    if len(added_noise_dots) > 0:
        noise_points = [(e, i) for i,e in enumerate(current_case) if case_classes[i] == 1 or int(i) in added_noise_dots]
    else:
        noise_points = [(e, i) for i,e in enumerate(current_case) if case_classes[i] == 1]
    print("noise_points")
    print(noise_points)
    if len(noise_points) < 2:
        return None

    if len(removed_noise_dots) > 0:
        noise_points = [a for a in noise_points if int(a[1]) not in removed_noise_dots]


    noise_points_values = [e[0] for e in noise_points]
    noise_points_indexes = [e[1] for e in noise_points]

    #mean noise is extracted
    mean_noise = np.mean(noise_points_values)
    std_noise = np.std(noise_points_values)
    max_noise = np.max(noise_points_values)

    #percentage of noise and peaks is extracted
    peak_freq = len(non_noise_points) / len(current_case)
    noise_freq = len(noise_points) / len(current_case)
    peak_to_noise_ratio = len(non_noise_points) / len(noise_points)

    #get noise areas
    noise_area = False
    noise_areas = []
    for i, e in enumerate(current_case):
        if i in noise_points_indexes and noise_area == True:
            noise_areas[-1].append(i)
        elif i in noise_points_indexes and noise_area == False:
            noise_areas.append([])
            noise_areas[-1].append(i)
            noise_area = True
        else:
            noise_area = False

    #get mean noise area size
    mean_noise_area_size = np.mean([len(a) for a in noise_areas])

    filtered_maxfilter_areas = []
    if filter_noise_area == True:
        #filter noise areas by mean excluding first and last areas
        filtered_maxfilter_areas = [noise_areas[0]]
        filtered_maxfilter_areas.extend([a for a in noise_areas[1:-1] if len(a) > mean_noise_area_size])
        filtered_maxfilter_areas.append(noise_areas[-1])
    else:
        filtered_maxfilter_areas = noise_areas.copy()

    #plot filtered noise areas
    filtered_maxfilter_indexes = []
    for a in filtered_maxfilter_areas:
        filtered_maxfilter_indexes.extend(a)

    filtered_maxfilter_values = [current_case[i] for i in filtered_maxfilter_indexes]

    #max noise is extracted
    max_filtered_noise = np.max(filtered_maxfilter_values)
    print("max_noise")
    print(max_noise)
    return non_noise_points, non_noise_points_values, non_noise_points_indexes, noise_points, noise_points_values, noise_points_indexes, mean_noise, std_noise, max_noise, peak_freq, noise_freq, peak_to_noise_ratio, noise_areas, mean_noise_area_size, filtered_maxfilter_areas, filtered_maxfilter_indexes, filtered_maxfilter_values, max_filtered_noise, cutoff_val


# def peak_detection(current_case, filter_noise_area=True, delta=False, stop_condition_perc=False, added_noise_dots=[], removed_noise_dots=[], cutoff_val=0.90):
def peak_detection(current_case, original_case=False, delta=False, expconfigs=[], stop_condition_perc=False, nargs=[]):
    non_noise_points, non_noise_points_values, non_noise_points_indexes, noise_points, noise_points_values, noise_points_indexes, mean_noise, std_noise, max_noise, peak_freq, noise_freq, peak_to_noise_ratio, noise_areas, mean_noise_area_size, filtered_maxfilter_areas, filtered_maxfilter_indexes, filtered_maxfilter_values, max_filtered_noise, cutoff_val = nargs
    #0                  1                       2                           3               4                   5                       6           7          8        9       10              11                  12              13                  14                      15                      16                  17
    # non_noise_points, non_noise_points_values, non_noise_points_indexes, noise_points, noise_points_values, noise_points_indexes, mean_noise, std_noise, max_noise, peak_freq, noise_freq, peak_to_noise_ratio, noise_areas, mean_noise_area_size, filtered_maxfilter_areas, filtered_maxfilter_indexes, filtered_maxfilter_values, max_filtered_noise = noise_detection(current_case, filter_noise_area=filter_noise_area, added_noise_dots=added_noise_dots, removed_noise_dots=removed_noise_dots, cutoff_val=cutoff_val)
    # print("non_noise_points")
    # print(non_noise_points)
    #delta is calculated based on mean of peak points divided by 3
    
    if len(expconfigs) > 1:
        endnoisecriteria = expconfigs[0]
        smoothbeforeregression = expconfigs[1]
        noiseratio = expconfigs[2]
        local_minimum_check = expconfigs[3]
        pass
    else:
        endnoisecriteria = 0.9
        # smoothbeforeregression = "noisecriteria"
        smoothbeforeregression = "never"
        noiseratio = 1.0
        # local_minimum_check = True
        local_minimum_check = False

    if delta == False:
        delta = np.mean(non_noise_points_values) / 3
        delta = float("{:.3f}".format(delta))

    # delta is average from detected maximum values to closest baseline minimum / 2
    maxtab, mintab = peakdet(current_case, delta)

    maxtab = [i for i in maxtab if i not in noise_points_indexes]

    #DEBUGGING
    # print("maxtab")
    # print(maxtab)
    # print("mintab")
    # print(mintab)

    # mintab = [i for i in mintab if i not in noise_points_indexes]

    above_before_derivatives = generate_bfderivative_full(current_case)

    all_local_maximums = []
    all_local_minimums = []
    for i in range(len(above_before_derivatives)-1):
        val = above_before_derivatives[i]
        val2 = above_before_derivatives[i+1]
        if val > 0.0 and val2 < 0.0:
            all_local_maximums.append(i)
        elif val < 0.0 and val2 > 0.0:
            all_local_minimums.append(i)
    #add first and last points to minimums if below max noise threshold
    if current_case[0] < np.max(noise_points_values) and current_case[1] > current_case[0]:
        all_local_minimums.insert(0, 0)
    if current_case[-1] < np.max(noise_points_values) and current_case[-2] > current_case[-1]:
        all_local_minimums.append(len(current_case)-1)
    
    #pick pairs of maximums, filter maximums after pairs if baseline not found

    #filter maximums by derivative
    maxtab = sorted(list(set(maxtab) & set(all_local_maximums)))
    
    f_points = []
    s_f_points = []
    t_points = []
    l_points = []
    exponential_pops = []

    i = 0
    while (True and len(maxtab) > 0):
        # = current_case[maxtab[i]]
        max_1_i = maxtab[i]

        f_point = None
        previous_mins = sorted(list(set([i for i,e in enumerate(current_case[:max_1_i])]) & set(all_local_minimums) & set(noise_points_indexes)))


        #max_2, max_2_i = None, None
        try:
            #max_2 = current_case[maxtab[i+1]]
            max_2_i = maxtab[i+1]
        except IndexError as e:
            print(e)
            break

        range_maxfilter = []
        try:
            #noise_above = [a for a in noise_points_indexes if a > max_2_i][0]
            filtered_maxfilter_area_above = [a for a in filtered_maxfilter_areas if a[0] > max_2_i][0]
            filtered_maxfilter_area_above_start = filtered_maxfilter_area_above[0]
            filtered_maxfilter_area_above_middle_p = int(np.median(filtered_maxfilter_area_above))
            filtered_maxfilter_area_endpoint = int(np.quantile(filtered_maxfilter_area_above, endnoisecriteria))
            #filtered_maxfilter_area_endpoint_values = current_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint]
            range_maxfilter = range(filtered_maxfilter_area_above_start,filtered_maxfilter_area_endpoint)
            # filtered_maxfilter_area_minmax = [i for i in range(filtered_maxfilter_area_above_start,filtered_maxfilter_area_endpoint) if i in all_local_minimums or i in all_local_maximums]
            filtered_maxfilter_area_minmax = [i for i in range_maxfilter if i in all_local_minimums or i in all_local_maximums]
            filtered_maxfilter_area_minmax_perc = len(filtered_maxfilter_area_minmax) / len(filtered_maxfilter_values)
        except IndexError as e:
            print(e)
            break

        after_mins = current_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint]
        #exponential regression done here

        valuesfit = None
        auto_mode = False
        if stop_condition_perc == False:
            auto_mode = True
            # stop_condition_perc = 0.05
            stop_condition_perc = 0.35
            # stop_condition_perc = 0.15

        ovaluesfit = None
        if original_case:
            ovaluesfit = original_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint]
        
        valuesfit = after_mins
        if smoothbeforeregression == "always":
            print("always selected, smoothing going to be done")
            # smoothed_vals = smooth_data(current_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint], 2)
            smoothed_vals = smooth_data(valuesfit, 2)
            print(len(valuesfit))
            print(len(smoothed_vals))
            if len(valuesfit) != len(smoothed_vals):
                valuesfit = smoothed_vals[2:]
            else:
                valuesfit = smoothed_vals.copy()
            if original_case:
                osmoothed_vals = smooth_data(ovaluesfit, 2)
                if len(ovaluesfit) != len(osmoothed_vals):
                    ovaluesfit = osmoothed_vals[2:]
                else:
                    ovaluesfit = osmoothed_vals.copy()

            # valuesfit = smoothed_vals[2:-2]
            print(len(valuesfit))

        elif smoothbeforeregression == "noisecriteria":
            print("noise criteria selected, verifying smoothing")
            if filtered_maxfilter_area_minmax_perc > noiseratio:
                print("smoothing done")
                # smoothed_vals = smooth_data(current_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint], 2)
                smoothed_vals = smooth_data(valuesfit, 2)
                print(len(valuesfit))
                print(len(smoothed_vals))
                if len(valuesfit) != len(smoothed_vals):
                    valuesfit = smoothed_vals[2:]
                else:
                    valuesfit = smoothed_vals.copy()
                if original_case:
                    osmoothed_vals = smooth_data(ovaluesfit, 2)
                    if len(ovaluesfit) != len(osmoothed_vals):
                        ovaluesfit = osmoothed_vals[2:]
                    else:
                        ovaluesfit = osmoothed_vals.copy()
            elif auto_mode == True:
                # stop_condition_perc = 0.01
                # stop_condition_perc = 0.95
                # stop_condition_perc = 0.20
                stop_condition_perc = 0.35
        elif smoothbeforeregression == "never":
            print("never selected, skipping smoothing")
            pass

        #TODO MORE HERE
        # if filtered_maxfilter_area_minmax_perc > 1.0:
        #     smoothed_vals = smooth_data(current_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint], 2)
        #     valuesfit = smoothed_vals[2:]
        # else:
        #     if auto_mode == True:
        #         stop_condition_perc = 0.01
        #     valuesfit = after_mins

        #Sanity check
        after_point = None
        if len(valuesfit) == 0:
            print("no values available for exponential inference (insufficient)")
            i += 1
            continue
        elif len(valuesfit) < 2:
            print("Single value available for exponential inference (insufficient)")
            print("afterpoint is single value")
            after_point = range_maxfilter[valuesfit.index(np.min(valuesfit))]
        elif len(valuesfit) > 2:
            print("multple values available for exponential inference")

            #x values created for real data and for high definition data
            valuesfitx = np.array(range_maxfilter)
            valuesfitx_highdef = np.linspace(max_2_i, filtered_maxfilter_area_endpoint, 100)

            #curve fit for exponential function. high def data appended as well
            print("try exponential fit")
            popt, pcov = curve_fit(exponential_fit, valuesfitx, valuesfit, p0 = (1e-6, 1e-6, 1), maxfev=150000)
            opopt, opcov = None, None
            if original_case:
                opopt, opcov = curve_fit(exponential_fit, valuesfitx, ovaluesfit, p0 = (1e-6, 1e-6, 1), maxfev=150000)
                opopt[0] = popt[0]
                opopt[1] = popt[1]
            
            print("exponential fit not found")
            exponential_pops.append((valuesfitx_highdef, popt))

            #integral of values for exponential are done
            total_area = 0.0
            if original_case:
                total_area = np.trapz([exponential_fit(point, *opopt) for point in range_maxfilter])
            else:
                total_area = np.trapz([exponential_fit(point, *popt) for point in range_maxfilter])
            print("range_maxfilter")
            print(range_maxfilter)
            print("total_area")
            print(total_area)
            current_area = 0.0
            print("stop_condition_perc")
            print(stop_condition_perc)
            stop_percentual = total_area * stop_condition_perc
            print("stop_percentual")
            print(stop_percentual)
            print("popt")
            print(popt)
            print("opopt")
            print(opopt)
            print("")
            print("")
            for point in range_maxfilter:
                print("iteration")
                after_point = point
                print("point")
                print(point)
                if original_case:
                    exponential_speed_value = exponential_fit(point, *opopt)
                else:
                    exponential_speed_value = exponential_fit(point, *popt)
                current_area += exponential_speed_value
                # range_summation = range(range_maxfilter[0], point+1)
                # print("range_summation")
                # print(range_summation)
                # current_area = np.trapz([exponential_fit(a, *popt) for a in range_summation])
                print("current_area")
                print(current_area)
                is_local_minimum = point in all_local_minimums
                if current_area >= stop_percentual and local_minimum_check == False:
                    break
                elif current_area >= stop_percentual and local_minimum_check == True and is_local_minimum == True:
                    break
            print("")
            print("")
            print("last point has been defined as: " + str(after_point) + " for perc == " + str(current_area))
        if after_point is None:
            i += 1
            continue


        # valuesfitx = np.array(range(filtered_maxfilter_area_above_start,filtered_maxfilter_area_endpoint))
        # valuesfitx_highdef = np.linspace(max_2_i, filtered_maxfilter_area_endpoint, 100)

        # after_point = None
        # if len(valuesfit) > 2:
        #     popt, pcov = curve_fit(exponential_fit, valuesfitx, valuesfit, p0 = (1e-6, 1e-6, 1), maxfev=150000)
        #     exponential_pops.append((valuesfitx_highdef, popt))
        #     # plt.plot(valuesfitx_highdef , exponential_fit(valuesfitx_highdef, *popt), color="purple")

        #     after_point = filtered_maxfilter_area_above_start-1
        #     prev_new_y_value = exponential_fit(after_point, *popt)
        #     for j in range(after_point+1, filtered_maxfilter_area_endpoint):
        #         after_point = j
        #         new_y_value = exponential_fit(after_point, *popt)
        #         ratio_new_old = 1.0 - (new_y_value / prev_new_y_value)
        #         # if (new_y_value <= mean_noise or ratio_new_old <= 0.05) and j in all_local_minimums and new_y_value < max_filtered_noise:
        #         # if ratio_new_old <= stop_condition_perc and j in all_local_minimums and new_y_value < max_filtered_noise:
        #         if ratio_new_old <= stop_condition_perc and j in all_local_minimums:
        #         # if new_y_value <= mean_noise and j in all_local_minimums:
        #             break
        #         prev_new_y_value = new_y_value
        # elif len(valuesfit) >= 1:
        #     after_point = range(filtered_maxfilter_area_above_start,filtered_maxfilter_area_endpoint)[valuesfit.index(np.min(valuesfit))]
        # else:
        #     i += 1
        #     continue
        
        if len(previous_mins) >= 1:
            f_point = previous_mins[-1]
            f_points.append(f_point)
        else:
            i += 1
            continue
        l_points.append(after_point)

        try:
            #maximum minimum refinement after f_point and l_point detection
            in_between = range(f_points[-1],l_points[-1]+1)
            in_between_maximums = sorted(list(set(in_between) & set(maxtab)))
            #filter maximum and minimum points if more than two between first and last points:
            in_between_maximums_vals_sort = sorted([(i,current_case[i]) for i in in_between_maximums], key=lambda x: x[1], reverse=True)
            #maximums are two highest maximums
            second_point = in_between_maximums_vals_sort[0][0]
            s_f_points.append(second_point)
            fourth_point = in_between_maximums_vals_sort[1][0]
            s_f_points.append(fourth_point)
            second_point = np.min([s_f_points[-1], s_f_points[-2]])
            fourth_point = np.max([s_f_points[-1], s_f_points[-2]])
            between_defined_maximums = [e for i,e in enumerate(current_case) if i > second_point and i < fourth_point]
            all_between_maxes = [(i+second_point+1, e) for i,e in enumerate(between_defined_maximums)]
            all_between_maxes_sort = sorted(all_between_maxes, key=lambda x: x[1], reverse=True)
            t_points.append(all_between_maxes_sort[-1][0])
        except IndexError:
            i += 1
            continue

        maxtab_skip_point = sorted([m for m in maxtab if m > filtered_maxfilter_area_above_middle_p])

        if len(maxtab_skip_point) > 0:
            i = maxtab.index(maxtab_skip_point[0])
        else:
            break


    return (f_points, s_f_points, t_points, l_points), (mean_noise, std_noise, max_noise, filtered_maxfilter_indexes), exponential_pops, (delta, stop_condition_perc, cutoff_val)
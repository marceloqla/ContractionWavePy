import math
import matplotlib.pyplot as plt
import numpy as np

#Linear Regression Peak fitting algorithm

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def noise_signal_linear_regression(current_case):
    try:
        #linear regression on first linear_range points
        linear_range = np.max([math.ceil(len(current_case)*0.05), 5])
        xf5 = range(linear_range)
        yf5 = sorted(current_case)[0:linear_range]
        Af5 = np.vstack([xf5, np.ones(len(xf5))]).T
        mf5, cf5 = np.linalg.lstsq(Af5, yf5)[0]
        ftoplot = [(mf5 * ex) + cf5 for ex in range(len(current_case))]

        #linear regression on last linear_range points
        xl5 = range(len(current_case)-linear_range,len(current_case))
        yl5 = sorted(current_case)[-linear_range:]
        Al5 = np.vstack([xl5, np.ones(len(xl5))]).T
        ml5, cl5 = np.linalg.lstsq(Al5, yl5)[0]
        ltoplot = [((ml5 * ex) + cl5) for ex in range(len(current_case))]

        #generate lines
        line1 = [[i, a] for i, a in enumerate(ftoplot[0:2])]
        line2 = [[i, a] for i, a in enumerate(ltoplot[0:2])]

        #calculate and plot intersection
        xintersect, yintersect = line_intersection(line1, line2)
        print("xintersect, yintersect")
        print(xintersect, yintersect)
        #test intersection for being above line. if so first regression skips more 5 points until intersect is below
        print("xintersect_dot")
        xintersect_dot = math.ceil(xintersect)
        print(xintersect_dot)
        yintersect_value = sorted(current_case)[xintersect_dot]
        skip_dots = 1

        while (yintersect_value < yintersect):
            linear_range = np.max([math.ceil(len(current_case)*0.05), 5])
            # condition for yielding non possible regression (poor value to noise ratio)
            if linear_range+skip_dots >= len(current_case) - linear_range:
                print("#POOR NOISE TO SIGNAL RATIO")
                return False
            xf5 = range(skip_dots,linear_range+skip_dots)
            yf5 = sorted(current_case)[skip_dots:linear_range+skip_dots]

            Af5 = np.vstack([xf5, np.ones(len(xf5))]).T
            mf5, cf5 = np.linalg.lstsq(Af5, yf5)[0]
            
            ftoplot = [(mf5 * ex) + cf5 for ex in range(len(current_case))]
            line1 = [[i, a] for i, a in enumerate(ftoplot[0:2])]
            line2 = [[i, a] for i, a in enumerate(ltoplot[0:2])]
            xintersect, yintersect = line_intersection(line1, line2)
            xintersect_dot = math.ceil(xintersect)
            yintersect_value = sorted(current_case)[xintersect_dot]
            skip_dots += 1
        return ftoplot, ltoplot, xintersect, yintersect, xintersect_dot, yintersect_value
    except Exception as e:
        print("Warning exception")
        print(e)
        return False

def generate_bfderivative_full(data, indexes=False):
    #generate before derivatives for all dots (whether diff between current dot and previous dot values is pos or neg)
    
    before_derivatives = []
    if not indexes:
        before_derivatives.append(0)

    for i in range(1, len(data)):
        delta = data[i] - data[i-1]
        if indexes and i in indexes:
            before_derivatives.append(delta)
        elif not indexes:
            before_derivatives.append(delta)
    return before_derivatives

def peakthresholding(current_case, plotting=False, noisedetection=False):
    if plotting == True:
        fig = plt.figure()
        plt.plot(sorted(current_case), color="red")

    # window_size = 2
    # averaged_data = smooth_data(current_case, window_size)
    noise_signal_results = noise_signal_linear_regression(current_case)
    if len(noise_signal_results) == 1:
        if noisedetection == True:
            return False
        else:
            print("Exiting due to impossible peak detection...")
            quit()
    ftoplot, ltoplot, xintersect, yintersect, xintersect_dot, yintersect_value = noise_signal_results


    #plot sorted data vs linear regression values
    if plotting == True:
        print("plotting linear regression data")
        ax_list = fig.axes
        ax_list[0].set_ylim([0.0,  math.ceil(np.max(current_case)) ])
        plt.plot(ftoplot, color="blue")
        plt.plot(ltoplot, color="blue")

        #plot intersecction between noise and signal linear regressions
        plt.scatter([xintersect], [yintersect], c="blue", alpha=0.5)
        plt.show()

    #get all dots above linear regression interseccion Y value (signal threshold)
    abovethresholddots = []
    abovethresholddots.extend([current_case.index(e) for e in sorted(current_case) if e >= yintersect_value])

    belowthresholddots = []
    belowthresholddots.extend([current_case.index(e) for e in sorted(current_case) if e < yintersect_value])
    
    above_before_derivatives = generate_bfderivative_full(current_case, abovethresholddots)
    
    abovethresholddots_localmaximums = []    
    for i in range(len(above_before_derivatives)-1):
        val = above_before_derivatives[i]
        val2 = above_before_derivatives[i+1]
        if val > 0.0 and val2 < 0.0:
            abovethresholddots_localmaximums.append(abovethresholddots[i])

    below_before_derivatives = generate_bfderivative_full(current_case, belowthresholddots)

    positive_below_before_derivatives = [abs(deriv) for deriv in below_before_derivatives]
    average_pos_below_bef_derivatives = np.mean(positive_below_before_derivatives)
    average_belowthresholddots = np.mean([e for i,e in enumerate(current_case) if i in belowthresholddots])
    # average_belowthresholddots = np.max([e for i,e in enumerate(current_case) if i in belowthresholddots])
    
    sum_averages_upper = average_belowthresholddots + average_pos_below_bef_derivatives
    sum_averages_lower = average_belowthresholddots - average_pos_below_bef_derivatives

    return average_belowthresholddots, abovethresholddots, sum_averages_upper, sum_averages_lower, abovethresholddots_localmaximums

def peakareadetection(current_case, average_belowthresholddots, abovethresholddots,sum_averages_upper, sum_averages_lower, plotting=False):
    peak = False
    current_peak = []
    valid_peaks = []
    test_valid_peaks = []
    #peak areas first detection
    for i, e in enumerate(current_case):
        if e > average_belowthresholddots:
            peak = True
        if e < average_belowthresholddots and peak == True:
            peak = False
        if peak == True:
            current_peak.append(i)
        else:
            if len(current_peak) > 0:
                if len(list(set(current_peak).intersection(abovethresholddots))) > 0:
                    valid_peaks.append(current_peak)
                    test_valid_peaks.extend([sum_averages_upper for a in current_peak])
                else:
                    test_valid_peaks.extend([sum_averages_lower for a in current_peak])
            current_peak = []
            test_valid_peaks.append(sum_averages_lower)

    if len(current_peak) > 0:
        if len(list(set(current_peak).intersection(abovethresholddots))) > 1:
            valid_peaks.append([current_case[i] for i in current_peak])
            test_valid_peaks.extend([sum_averages_upper for a in current_peak])
        else:
            test_valid_peaks.extend([sum_averages_lower for a in current_peak])
    
    if plotting == True:
        print("plotting peak area detection")
        plt.figure()
        plt.plot(current_case, color="red")
        plt.plot(test_valid_peaks, color="green")
        plt.show()
    return valid_peaks
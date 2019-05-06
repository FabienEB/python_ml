from statsmodels.stats.power import  tt_ind_solve_power
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def test_ttest_power_diff(mean, stdev, sample1_size=None, alpha=0.05, desired_power=0.8, mean_diff_percentages=[0.05, 0.15]):


    # Notes 

    # The higher mean, stdev, desired power the greaters the sample sized needed. The higher the mean difference the lower the required.
    # mean : the desired mean of the test group 
    # stdev : standard deviation, Cohens D effect size = 0.2 'small', 0.5 'medium', +0.8 'large'. 
    #     This means that if two groups' means don't differ by 0.2 standard deviations or more, the difference is trivial, even if it is statistically signficant.
    # sample1_size : is NONE then sample is 1000 and assumes equal sample on both sides 
    # desidered_power : aet at 0.8, as per Cohens recom 
    # mean_diff_percentages : A line will be plotted per number

    fig, ax = plt.subplots()
    for mean_diff_percent in mean_diff_percentages:
        mean_diff = mean_diff_percent * mean
        effect_size = mean_diff / stdev

        print('Mean diff: ', mean_diff)
        print('Effect size: ', effect_size)

        powers = []

        max_size  = sample1_size
        if sample1_size is None:
            max_size = 1000

        sizes = np.arange(1, max_size, 2)
        for sample2_size in sizes:
            if(sample1_size is None):
                n = tt_ind_solve_power(effect_size=effect_size, nobs1=sample2_size, alpha=alpha, ratio=1.0, alternative='two-sided')
                print('tt_ind_solve_power(alpha=', alpha, 'sample2_size=', sample2_size, '): sample size in *second* group: {:.5f}'.format(n))
            else:
                n = tt_ind_solve_power(effect_size=effect_size, nobs1=sample1_size, alpha=alpha, ratio=(1.0*sample2_size/sample1_size), alternative='two-sided')
                print('tt_ind_solve_power(alpha=', alpha, 'sample2_size=', sample2_size, '): sample size *each* group: {:.5f}'.format(n))

            powers.append(n)

        try: # mark the desired power on the graph
            z1 = interp1d(powers, sizes)
            results = z1(desired_power)

            plt.plot([results], [desired_power], 'gD')
        except Exception as e:
            print("Error: ", e)
            #ignore

        plt.title('Power vs. Sample Size')
        plt.xlabel('Sample Size')
        plt.ylabel('Power')

        plt.plot(sizes, powers, label='diff={:2.0f}%'.format(100*mean_diff_percent)) #, '-gD')

    plt.legend()
    plt.show()
    
  
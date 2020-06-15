#-------------------------------FUNCTIONS-------------------------------------#
#
# 1.data_description(data_with_labels,data_without_labels,data)
#     input : 3 dataframes: data together with their labels, without them, initial data with bytes
#     output: print a description in the console
#
# 2.plotPerColumnDistribution(df, nGraphShown, nGraphPerRow)
#     input : dataframe you want to visualize,modify the size of the plot
#     output: several plots
#
# 3.plotCorrelationMatrix(df, graphWidth)
#     input : dataframe you want to visualize,modify the graph width
#     output: several plots
#
# 4.plotScatterMatrix(df, plotSize, textSize)
#     input : dataframe you want to visualize,modify the sizes of plot and text
#     output: several plots
# 5.data_visualization_control(df) - use this to control the other 3 funvtions
#     input : dataframe you want to visualize
#     output: several plots
#
# 6.cooccurrence_matrix(labels,categories,name)
#     input : labels , label names , a title
#     output: 2 plots, one for co-occurrence matrix and one for co-occurrence matrix percentage
#
# 7.get_best_distribution(data)
#     input : numerical values
#     output: find best distribution from: "norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"
#
# 8.find_feature_and_label_dist2(data_with_labels,dataset_features)
#     input : whole dataset features and labels, number of features
#     output: 2 lists with distributions for features and for labels
#
# 9.best_fit_distribution(data, bins=200, ax=None)
#     input : whole dataset features and labels, number of features
#     output: best fit after a search of 11 distributions
#
# 10.make_pdf(dist, params, size=10000)
#     input : distribution
#     output: pdf
#
# 11.find_best_distribution(data,title,xtitle,ytitle)
#     input : feature , strings for plot : title,xtitle,ytitle
#     output: 2 plots for each feature, one with all the distributions, one with the best fit
#
# 12.find_feature_and_label_dist(data_with_labels,dataset_features)
#     input : whole dataset features and labels, number of features
#     output: 2 plots for each feature, one with all the distributions, one with the best fit
#
#-----------------------------------------------------------------------------#
#--------------------------IMPORT LIBRARIES-----------------------------------#
import time,os
from os import path
import numpy as np 
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import warnings
import matplotlib.pyplot as plt
from cycler import cycler
#-----------------------------------------------------------------------------#
#-------------------------------FUNCTIONS-------------------------------------#
def data_description(data_with_labels,data_without_labels,data):
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    
    print(data_with_labels.describe(include='all').transpose())
    print(data_without_labels.describe(include='all').transpose())
    print(data.describe(include=['object']))

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    #nunique = df.nunique()
    # For displaying purposes, pick columns that have between 1 and 50 unique values
    #df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] 
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    #plt.figure()
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})', fontsize=10,verticalalignment='top')        
#        # fix for mpl bug that cuts off top/bottom of seaborn viz
#        b, t = plt.ylim() # discover the values for bottom and top
#        b += 0.5 # Add 0.5 to the bottom
#        t -= 0.5 # Subtract 0.5 from the top
#        plt.ylim(b, t) # update the ylim(bottom, top) values        
    plt.subplots_adjust(wspace = 0.2,hspace = 0.2)    
    #plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

def plotCorrelationMatrix(df, graphWidth):
    #filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    # keep columns where there are more than 1 unique values
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    #plt.figure()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    #plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    #plt.figure()
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    #plt.set_xticklabels(rotation=90, ha='right')
    #plt.xticks(rotation=90)
    #plt.yticks(rotation=90)
    plt.show()
    
def data_visualization_control(df):
    # Distribution graphs (histogram/bar graph) of column data
    #df=data_test
    nGraphShown=12  # Number of features displayed
    nGraphPerRow=6 # Number of rows for the graph displayed
    plotPerColumnDistribution(df, nGraphShown, nGraphPerRow)
#    sec = input('Enter how many seconds to sleep until next plot.\n')
#    print('Going to sleep for', sec, 'seconds.')
#    time.sleep(int(sec))
    time.sleep(10)
    plt.close()
    # Correlation matrix
    graphWidth=15
    plotCorrelationMatrix(df, graphWidth)
#    sec = input('Enter how many seconds to sleep until next plot.\n')
#    print('Going to sleep for', sec, 'seconds.')
#    time.sleep(int(sec))
    time.sleep(10)
    plt.close()
    # Scatter and density plots
    plotSize=8
    textSize=6
    plotScatterMatrix(df, plotSize, textSize)
#    sec = input('Enter how many seconds to sleep until next plot.\n')
#    print('Going to sleep for', sec, 'seconds.')
#    time.sleep(int(sec))
    time.sleep(10)
    plt.close()

# cooccurrence matrix 
def cooccurrence_matrix(labels,categories,name):
    # Compute cooccurrence matrix 
    cooccurrence_matrix = np.dot(labels.transpose(),labels)
    print('\ncooccurrence_matrix:\n{0}'.format(cooccurrence_matrix)) 
    # Compute cooccurrence matrix in percentage
    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))
    print('\ncooccurrence_matrix_percentage:\n{0}'.format(cooccurrence_matrix_percentage))
    
    df_cm = pd.DataFrame(cooccurrence_matrix, index=categories, columns=categories )
    fig = plt.figure(figsize = (15,8))
    plt.title('Co-occurrence Matrix for ' + name+'.') 
    Ηeatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='OrRd',linewidths=1,linecolor='k',square=True,mask=False,cbar_kws={"orientation": "vertical"},cbar=True)
#    # fix for mpl bug that cuts off top/bottom of seaborn viz
#    b, t = plt.ylim() # discover the values for bottom and top
#    b += 0.5 # Add 0.5 to the bottom
#    t -= 0.5 # Subtract 0.5 from the top
#    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.xticks(rotation=0,fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.show()
    
    df_cm_per = pd.DataFrame(cooccurrence_matrix_percentage, index=categories, columns=categories )
    fig = plt.figure(figsize = (15,15))
    plt.title('Co-occurrence Matrix percentage for ' + name+'.') 
    Ηeatmap = sns.heatmap(df_cm_per, annot=True, fmt="f",annot_kws={"fontsize":"small"},cmap='OrRd',linewidths=1,linecolor='k',square=True,mask=False,cbar_kws={"orientation": "vertical"},cbar=True)
#    # fix for mpl bug that cuts off top/bottom of seaborn viz
#    b, t = plt.ylim() # discover the values for bottom and top
#    b += 0.5 # Add 0.5 to the bottom
#    t -= 0.5 # Subtract 0.5 from the top
#    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.xticks(rotation=0,fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.show()
    
def get_best_distribution(data):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

#    print("Best fitting distribution: "+str(best_dist))
#    print("Best p value: "+ str(best_p))
#    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

def find_feature_and_label_dist2(data_with_labels,dataset_features):
    feature_distributions=[]
    label_distributions=[]
    for i in range(data_with_labels.shape[1]):
        if i <= dataset_features:
            print('Distribution for feature' +str(i))
            best_dist_feature,_,_=get_best_distribution(data_with_labels.iloc[:,i])
            feature_distributions.append(best_dist_feature)
        if i > dataset_features:
            print('Distribution for label' +str(i-dataset_features))
            best_dist_label,_,_=get_best_distribution(data_with_labels.iloc[:,i])
            label_distributions.append(best_dist_label)
    return(feature_distributions,label_distributions)
    
# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    # for 89 distributions
#    DISTRIBUTIONS = [        
#        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
#        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
#        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
#        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
#        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
#        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
#        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
#        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
#        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
#        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
#    ]
#    # for 34 distributions
#    DISTRIBUTIONS = [        
#        st.alpha,st.beta,st.cauchy,st.chi,st.chi2,st.cosine,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,
#        st.genlogistic,st.genpareto,st.gennorm,st.genexpon,st.gamma,st.halfcauchy,st.halflogistic,st.halfnorm,st.invgauss,
#        st.laplace,st.logistic,st.lognorm,st.lomax,st.norm,st.pareto,st.pearson3,st.powerlaw,st.rdist,
#        st.rayleigh,st.rice,st.uniform,st.weibull_min,st.weibull_max
#    ]
    # for 11 distributions
    DISTRIBUTIONS = [        
        st.beta,st.cauchy,st.expon,st.gamma,st.laplace,st.norm,st.pareto,st.pearson3,st.rayleigh,st.rice,st.uniform
    ]
    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf
    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # fit dist to data
                params = distribution.fit(data)
                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
        except Exception:
            pass
    return(best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)
    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return(pdf)

def find_best_distribution(data,dirname,title,xtitle,ytitle):
    matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
    #matplotlib.style.use('ggplot')
    matplotlib.style.use('classic')
    matplotlib.rcParams['figure.max_open_warning'] = 0
    # Plot for comparison
    plt.figure(figsize=(12,8))
    #ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5)#, plt.rcParams['axes.prop_cycle']= cycler(color='bgrcmyk')
    ax.prop_cycle    : cycler('color', 'bgrcmyk')
    # Save plot limits
    dataYLim = ax.get_ylim()
    
    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)
    
    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(str(title)+'\n All Fitted Distributions')
    ax.set_xlabel(str(xtitle))
    ax.set_ylabel(str(ytitle))
    
    plt.savefig(dirname+ '/' +title+"_1",bbox_inches ='tight', pad_inches=0)
    plt.close()
    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)
    
    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
    
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    
    ax.set_title(str(title)+' with best fit distribution \n' + dist_str)
    ax.set_xlabel(str(xtitle))
    ax.set_ylabel(str(ytitle))
    plt.savefig(dirname+ '/' +title+"_2",bbox_inches ='tight', pad_inches=0)
    plt.close()
    return(dist_str,best_fit_name)
    
def find_feature_and_label_dist(data_with_labels,dataset_features,path_out):
    if path.exists(path_out):
        print('The '+path_out+' file already exists!')
    else:   
        feature_distributions=[]
        dirname='./'+path_out+'/'
        create_dir(dirname)
        for i in range(data_with_labels.shape[1]):
            if i < dataset_features:
                name_feature,best_fit_name=find_best_distribution(data_with_labels.iloc[:,i],dirname,'feature' +str(i),'number of instances','values')
                print('Best distribution for feature' +str(i)+' '+name_feature)
                feature_distributions.append(best_fit_name)
    #        if i > dataset_features:
    #            name_label=find_best_distribution(data_with_labels.iloc[:,i],'label' +str(i-dataset_features),'number of instances','values')
    #            print('Best distribution for label' +str(i-dataset_features)+name_label)
        return(feature_distributions)        
    
def create_dir(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)
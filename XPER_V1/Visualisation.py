# =============================================================================
#                           Packages
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc('text', usetex = True) # TeX 


# =============================================================================
#       Bar plot: X-axis = phi_j value or pct // y-axis = Feature names
# =============================================================================

def feature_imp(Contribution,data,labels,metric,nb_var=5,percentage=True,echantillon="train"):
    
    ecart_contrib = sum(Contribution) - Contribution[0]
                                      
    if percentage == True:       

        Contribution = (Contribution / ecart_contrib)*100
    
        print("Contribution (%) sum: ", sum(Contribution[1:]) )
        
    selection = pd.DataFrame(Contribution[1:],index=data.columns,columns=["Metric"])
    
    selection["Labels"] = labels
    
    selection["absolute"] = abs(selection["Metric"])
    
    testons = selection.sort_values(by="absolute",ascending=False)[:nb_var].index
    
    final_selection_values = selection.loc[testons,"Metric"].values
    final_selection_labels = selection.loc[testons,"Labels"]
    
    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    y_pos = np.arange(0,2*len(final_selection_values),2)
    
    ax.barh(y_pos, final_selection_values, align='center',color="#893F45")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(final_selection_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    if percentage == True:
        ax.set_xlabel('Contribution (%)')
    else:
        ax.set_xlabel('Contribution')
        
    sns.despine()
    plt.tight_layout()
    plt.axvline(x=0, color='black', ls='--', lw=1)
    #plt.savefig('./Figures/Global_ESHAP_XGB_{}_{}.eps'.format(metric,echantillon), format='eps', dpi = 1200, bbox_inches='tight')
    #plt.savefig('./Figures/Global_ESHAP_XGB_{}_{}.pdf'.format(metric,echantillon), format='pdf', dpi = 1200, bbox_inches='tight')
    plt.show()
    
  
    
# =============================================================================
#                                   Force plots
# =============================================================================
    
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


def draw_higher_lower_element(out_value, offset_text):
    plt.text(out_value - offset_text, 0.495, 'higher',
             fontsize=13, color='#FF0D57',
             horizontalalignment='right')

    plt.text(out_value + offset_text, 0.495, 'lower',
             fontsize=13, color='#1E88E5',
             horizontalalignment='left')
    
    plt.text(out_value, 0.49, r'$\leftarrow$',
             fontsize=13, color='#1E88E5',
             horizontalalignment='center')
    
    plt.text(out_value, 0.515, r'$\rightarrow$',
             fontsize=13, color='#FF0D57',
             horizontalalignment='center')
    
def force_plot(XPER,perf_value,base_value, X, X_names,savefig_name,figsize=(8,4),min_perc=0.00001):
    
    # On ne prend pas volontairement le 0 en compte car aucun apport d'info
    pos_values = np.array(sorted(XPER[XPER>0],reverse=True))
    neg_values = np.array(sorted(XPER[XPER<0],reverse=False))

    #####
    
    index_sort_pos = np.argsort(-XPER[XPER>0]) # signe - car on index ensuite par les variables les plus importantes

    
    index_sort_neg = np.argsort(XPER[XPER<0]) # signe - car ordonne par ordre croissant

    #####

    offset_text = (np.abs(pos_values.sum()) + np.abs(neg_values.sum())) * 0.01
    
    #print("Here",perf_value,pos_values.sum())
    x_min = perf_value - pos_values.sum()# axe sont inverses 
    x_max = perf_value + (-neg_values.sum())  # axe sont inverses 
        
    # Define plots
    fig, ax = plt.subplots(figsize=figsize)

    # Plot height
    height = 0.13
    base_height = 0.03

    # Plot limits
    plt.xlim([x_min-0.05,x_max+0.075])
    plt.ylim([-0.5,0.15])

    # Width separator 
    global width_separators
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    #for values in pos_values: # Iteration sur les XPER positives
    base_x = perf_value.copy()
    
    fleche = XPER.max() /30
    for i,values in enumerate(pos_values): # Iteration sur les XPER positives
    
        
        if i == 0:
            
            # Forme polygon
            cut_y = 0.08 # fix 
            cut_x = values - fleche # variables
                 
            ### Polygon de separation avant le vrai polygon 
            polygon_sep = plt.Polygon([
                           (base_x - cut_x - width_separators, cut_y),
                           (base_x - values - width_separators, height),
                           (base_x, height),
                           (base_x, base_height),
                           (base_x - values - width_separators, base_height),
                           (base_x - cut_x - width_separators, cut_y)
                        ], closed=True, fill=True,facecolor="#FFC3D5", linewidth=0) #  "#FFC3D5"
                        
            plt.gca().add_patch(polygon_sep)
                             
            polygon = plt.Polygon([
                           (base_x - cut_x, cut_y),
                           (base_x - values, height),
                           (base_x, height),
                           (base_x, base_height),
                           (base_x - values, base_height),
                           (base_x - cut_x, cut_y)
                        ], closed=True, fill=True,facecolor="#FF0D57", linewidth=0) #  "#FF0D57"
                        
            plt.gca().add_patch(polygon)
 
        else:
            
            # Forme polygon 
            cut_y = 0.08 # fix 
            cut_x = values - fleche  # variables
            
                      
                        ### Polygon de separation avant le vrai polygon 
            polygon_sep = plt.Polygon([
                        (ancien_cut_x - width_separators, cut_y),
                        (base_x - width_separators,height),
                        (base_x - values - width_separators, height),
                        (base_x - cut_x - width_separators, cut_y),
                        (base_x - values - width_separators , base_height),
                        (base_x - width_separators,base_height),
                        (ancien_cut_x- width_separators, cut_y)], 
                        closed=True, fill=True,facecolor="#FFC3D5", linewidth=0) #  "#FFC3D5"
                        
            plt.gca().add_patch(polygon_sep)
                   
            polygon = plt.Polygon([
                        (ancien_cut_x, cut_y),
                        (base_x,height),
                        (base_x - values, height),
                        (base_x - cut_x , cut_y),
                        (base_x - values , base_height),
                        (base_x ,base_height),
                        (ancien_cut_x, cut_y)],
                            closed=True, fill=True,facecolor="#FF0D57", linewidth=0) #  "#FF0D57"
            
            plt.gca().add_patch(polygon)    
        
        
        ancien_cut_x = base_x - cut_x - width_separators
        # Changement depart de l'axe des abscisses 
        base_x +=  - values - width_separators       
    
    fig,ax = draw_labels(fig=fig, ax=ax, perf_value=perf_value,features=pos_values, 
                                 feature_type="positive", X=X[XPER>0][index_sort_pos], X_names=X_names[XPER>0][index_sort_pos],
                        offset_text=offset_text, min_perc=min_perc, text_rotation=0)
           
    base_x = perf_value.copy()
    
    for i,values in enumerate(neg_values): # Iteration sur les XPER negatives
    
        values = -values 
        if i == 0:
            
            # Forme polygon
            cut_y = 0.08 # fix 
            cut_x = values - fleche # variables
                        
            ### Polygon de separation avant le vrai polygon 
            polygon_sep = plt.Polygon([
                           (base_x + cut_x + width_separators, cut_y),
                           (base_x + values + width_separators, height),
                           (base_x, height),
                           (base_x, base_height),
                           (base_x + values + width_separators, base_height),
                           (base_x + cut_x + width_separators, cut_y)
                        ], closed=True, fill=True,facecolor="#D1E6FA", linewidth=0) #  "#D1E6FA" 
                        
            plt.gca().add_patch(polygon_sep)
                                
            polygon = plt.Polygon([
                           (base_x + cut_x, cut_y),
                           (base_x + values, height),
                           (base_x, height),
                           (base_x, base_height),
                           (base_x + values, base_height),
                           (base_x + cut_x, cut_y)
                        ], closed=True, fill=True,facecolor="#1E88E5", linewidth=0) #  
                        
            plt.gca().add_patch(polygon)

        else:
            
            # Forme polygon 
            cut_y = 0.08 # fix 
            cut_x = values - fleche  # variables
            
                       
                        ### Polygon de separation avant le vrai polygon 
            polygon_sep = plt.Polygon([
                        (ancien_cut_x + width_separators, cut_y),
                        (base_x + width_separators,height),
                        (base_x + values + width_separators, height),
                        (base_x + cut_x + width_separators, cut_y),
                        (base_x + values + width_separators , base_height),
                        (base_x + width_separators,base_height),
                        (ancien_cut_x+ width_separators, cut_y)], 
                        closed=True, fill=True,facecolor="#D1E6FA", linewidth=0) # "#FFC3D5" "#D1E6FA"
                        
            plt.gca().add_patch(polygon_sep)
                   
            polygon = plt.Polygon([
                        (ancien_cut_x, cut_y),
                        (base_x,height),
                        (base_x + values, height),
                        (base_x + cut_x , cut_y),
                        (base_x + values , base_height),
                        (base_x ,base_height),
                        (ancien_cut_x, cut_y)],
                            closed=True, fill=True,facecolor="#1E88E5", linewidth=0) #  "#1E88E5"
            
            plt.gca().add_patch(polygon)    
        
        ancien_cut_x = base_x + cut_x + width_separators
        # Changement depart de l'axe des abscisses 
        base_x += values + width_separators   
    
    fig,ax = draw_labels(fig=fig, ax=ax, perf_value=perf_value,features=neg_values, 
                                 feature_type="negative", X=X[XPER<0][index_sort_neg], X_names=X_names[XPER<0][index_sort_neg],
                        offset_text=offset_text, min_perc=min_perc, text_rotation=0)
           
    ## Barre pour situe l'individual payoff   
    x, y = np.array([[base_value, base_value], [0.13, 0.33]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    # Texte pour montrer individual payoff
    text_out_val = plt.text(base_value, 0.38, 'Benchmark', # hauteur de 0.33
                                fontsize=14, alpha=0.5,
                                horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    ## Barre pour situe la performance individuelle
    x, y = np.array([[perf_value, perf_value], [0.13, 0.15]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    # Texte pour montrer individual payoff
    text_out_val_PM = plt.text(perf_value, 0.38, 'Performance', # hauteur de 0.33
                                fontsize=14, alpha=0.5,
                                horizontalalignment='center')
    text_out_val_PM.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    # Texte pour montrer individual payoff
    
    x, y = np.array([[perf_value, perf_value], [0.13, 0.26]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)
    
    text_out_val_PM = plt.text(perf_value, 0.29, '{}'.format(perf_value), # hauteur de 0.33
                                fontsize=16,
                                horizontalalignment='center', weight='bold')
    #text_out_val_PM.set_bbox(dict(facecolor='white', edgecolor='white'))
    
    draw_higher_lower_element(out_value=perf_value, offset_text=offset_text)
    
    
    # Ticks at the top of the graphic + at the top of the axis
    
    

    plt.tick_params(top=True, bottom=False, left=False, right=False, 
                    labelleft=False, labeltop=True, labelbottom=False)
    plt.locator_params(axis='x', nbins=12)
    

    # Remove axis except the top 
    for key, spine in zip(plt.gca().spines.keys(), plt.gca().spines.values()):
            if key != 'top':
                spine.set_visible(False)

    plt.savefig(f'./Figures/{savefig_name}.pdf', format='pdf', dpi = 1200)

    plt.show()  
        
    
##############################################################################

def draw_labels(fig, ax, perf_value, features, feature_type, X, X_names,
                offset_text, min_perc=0.5, text_rotation=0):
    
    start_text = perf_value
    pre_val = perf_value
    
    # Define variables specific to positive and negative effect features
    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5'] # ['#FF0D57', '#FFC3D5'] ['#FF0D57', '#FFC3D5']
        alignement = 'right'
        sign = 1
    else:
        colors =  ['#1E88E5', '#D1E6FA']# ['#1E88E5', '#D1E6FA']
        alignement = 'left'
        sign = -1
    
    # Draw initial line
    if feature_type == 'positive':
        x, y = np.array([[pre_val, pre_val], [0.03, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val
        
    #Features contribution
    feature_contribution = features / np.sum(features)#      np.abs(features/perf_value)
    #print(feature_contribution)
    #print(min_perc)
    #feature_contribution = feature_contribution[feature_contribution>min_perc]
    
    box_end = perf_value
    val = perf_value
    for i,feature in enumerate(features):
        # Exclude all labels that do not contribute at least min_perc to the total
        if feature_contribution[i] < min_perc:
            #print("break because of low contribution:",i)
            #print("feature contribution",feature_contribution[i])
            #print("spread",feature_contribution[i] < min_perc)
            break
        
        # Compute value for current feature
        val = float(val - feature -sign*width_separators)
        # Draw labels.
        text = X_names[i] + "=" + str(X[i])
        
        if text_rotation != 0:
            va_alignment = 'top'
        else:
            va_alignment = 'baseline'

        text_out_val = plt.text(start_text - sign * offset_text,
                                -0.15, text,
                                fontsize=12, color=colors[0],
                                horizontalalignment=alignement,
                                va=va_alignment,
                                rotation=text_rotation)
        text_out_val.set_bbox(dict(facecolor='none', edgecolor='none'))

        # We need to draw the plot to be able to get the size of the
        # text box
        fig.canvas.draw()
        box_size = text_out_val.get_bbox_patch().get_extents()\
                               .transformed(ax.transData.inverted())
                               
        if feature_type == 'positive':
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]
        
        # If the feature goes over the side of the plot, we remove that label
        # and stop drawing labels
        if box_end_ > ax.get_xlim()[1]:
            print("box_end",box_end_)
            text_out_val.remove()
            break
        
        # Create end line
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0.03, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end],
                             [0.03, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end
        
        # Update previous value
        pre_val = float(pre_val -feature -sign*width_separators)

    # Create line for labels
    extent_shading = [perf_value, box_end, 0.03, -0.31]
    path = [[perf_value, 0.03], [pre_val, 0.03], [box_end, -0.08],
            [box_end, -0.2], [perf_value, -0.2],
            [perf_value, 0.03]]
    
    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch) 
    
    # Extend axis if needed
    lower_lim, upper_lim = ax.get_xlim()
    if (box_end < lower_lim):
        ax.set_xlim(box_end, upper_lim)
    
    if (box_end > upper_lim):
        ax.set_xlim(lower_lim, box_end)
        
    # Create shading
    if feature_type == 'negative':
        colors =  np.array([(30, 136, 229), (255, 255, 255)]) / 255. # np.array([(255, 13, 87), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.  # np.array([(30, 136, 229), (255, 255, 255)]) / 255.
    
    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)
    
    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = plt.imshow(Z2, interpolation='quadric', cmap=cm,
                    vmax=0.01, alpha=0.3,
                    origin='lower', extent=extent_shading,
                    clip_path=patch, clip_on=True, aspect='auto')
    im.set_clip_path(patch)
    
    return fig, ax

# =============================================================================
#                           End of force plots function
# =============================================================================





# =============================================================================
#                       Prerequisites for the use of SHAP plots 
# =============================================================================

import xgboost
import shap

### Idea: We use SHAP on a given model and then we change the values, data, 
### feature_names, base_values to the one obtained with XPER. Therefore, we
### are able to use the shap plots from the shap package on our results from XPER.
### Question: Can we avoid this step and direcly create the same object with our 
###           XPER values to use all of SHAP plots with XPER?


# train XGBoost model
X_useless,y_useless = shap.datasets.adult()

X_useless = X_useless.iloc[:100,:]
y_useless = y_useless[:100]

model = xgboost.XGBClassifier().fit(X_useless, y_useless)

# compute SHAP values
explainer = shap.Explainer(model, X_useless)
shap_values = explainer(X_useless)   # This is the object of interests for which
                                      # we are going to the change the values to 
                                      # those of XPER.

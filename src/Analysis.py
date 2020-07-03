from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import torch
import random


class Analysis():
    def __init__(self):
        pass
    
     
    def useful_plots(self,confusion_matrices):
        n_sample = 1000
        classes_for_batch = 10
        previous_accuracies = [None]
        total_accuracies = []
        new_accuracies = []
        batches_accuracies = []
        gaps_old_classes = [None]
        gaps_new_classes = [None]
        gaps = []
        x_gaps = []
        n_classes = []

        for step in range(len(confusion_matrices)):
            count = 0
            sum_true_previous = 0
            sum_true_new = 0
            sum_batch = 0
            tot = n_sample *(step)
            current_matrix = confusion_matrices[step]
            batches_accuracies.append([])
            for el in range(step):
                batches_accuracies[step].append(None)

            for i in range(len(current_matrix)): 
                if step == 0:
                    sum_true_new += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count += 1

                if step > 0 and i < (len(current_matrix)-10):
                    sum_true_previous += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count += 1

                if step > 0 and i >= (len(current_matrix)-10):
                    sum_true_new += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count +=1

                if ((i+1) % classes_for_batch) == 0:
                    b = int(((i+1) / classes_for_batch)-1)
                    batches_accuracies[b].append((sum_batch)/n_sample)
                    sum_batch = 0


            new_accuracies.append(100*(sum_true_new/n_sample))
            total_accuracies.append(100*((sum_true_previous + sum_true_new)/(tot + n_sample)))
            n_classes.append(10*(step+1))
            
            if step > 0:
                previous_accuracies.append(100*(sum_true_previous/tot))
                #gaps_old_classes.append(total_accuracies[step - 1] - previous_accuracies[step])
                gaps_old_classes.append(100*(previous_accuracies[step] / total_accuracies[step - 1]))
                gaps.append(previous_accuracies[step])
                gaps.append(total_accuracies[step])
                x_gaps.append(10*(step+1))
                x_gaps.append(10*(step+1))
            else:
                gaps.append(total_accuracies[step])
                x_gaps.append(10*(step+1))

        for i in range(1, len(new_accuracies)):
            gaps_new_classes.append(100*(new_accuracies[i] / 100))

        fig, ax = plt.subplots(figsize=(6,5))

        ax.plot(n_classes, previous_accuracies, marker ='o', label='Accuracy old classes')
        ax.plot(n_classes, new_accuracies, marker ='o', label='Accuracy new classes')
        ax.plot(n_classes, total_accuracies, marker ='o', label='Total accuracy')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Accuracy (%)')
        plt.yticks(np.arange(0, 100, 10))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(6,5))

        ax.plot(n_classes, gaps_old_classes, marker ='o', label='Gap old classes')
        #ax.plot(n_classes, gaps_new_classes, marker ='o', label='Accuracy new classes')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Gap (%)')
        #plt.yticks(np.arange(0, 11, 1))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(6,5))

        ax.plot(x_gaps, gaps, marker ='o', label='Gap')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Gap (%)')
        #plt.yticks(np.arange(0, 11, 1))
        plt.xticks(np.arange(0, 100, 20))

        plt.tight_layout()
        plt.grid()
        plt.show()


        fig, ax = plt.subplots(figsize=(12,10))

        for i in range(1):#len(batches_accuracies)):
            ax.plot(n_classes, batches_accuracies[i], marker ='o', label = f'Accuracy batch {i+1}')

        ax.legend()
        plt.xlabel('N Classes')
        plt.ylabel('Accuracy (%)')
        major_ticks = np.arange(0, 1.1, 0.1)
        minor_ticks = np.arange(0, 1, 0.02)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(10,110,10))
        ax.set_xlim(xmin=9,xmax=101)
        ax.set_ylim(ymin=0.35,ymax=1.0)

        plt.tight_layout()
        plt.grid(axis='y')
        plt.show()

    def getPrevNewAccuracies(self,confusion_matrices):
        n_sample = 1000
        classes_for_batch = 10
        previous_accuracies = [None]
        total_accuracies = []
        new_accuracies = []
        batches_accuracies = []
        n_classes = []

        for step in range(len(confusion_matrices)):
            count = 0
            sum_true_previous = 0
            sum_true_new = 0
            sum_batch = 0
            tot = n_sample *(step)
            current_matrix = confusion_matrices[step]
            batches_accuracies.append([])
            for _ in range(step):
                batches_accuracies[step].append(None)

            for i in range(len(current_matrix)): 
                if step == 0:
                    sum_true_new += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count += 1

                if step > 0 and i < (len(current_matrix)-10):
                    sum_true_previous += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count += 1

                if step > 0 and i >= (len(current_matrix)-10):
                    sum_true_new += current_matrix[i][i]
                    sum_batch += current_matrix[i][i]
                    count +=1

                if ((i+1) % classes_for_batch) == 0:
                    b = int(((i+1) / classes_for_batch)-1)
                    batches_accuracies[b].append((100*sum_batch)/n_sample)
                    sum_batch = 0


            new_accuracies.append((sum_true_new/n_sample))
            total_accuracies.append(100*((sum_true_previous + sum_true_new)/(tot + n_sample)))
            n_classes.append(10*(step+1))
            
            if step > 0:
                previous_accuracies.append((sum_true_previous/tot))
            
        return previous_accuracies,new_accuracies

    def plotConfMatrix(self,confusion_matrix,title,scale_log=False):
        confusion_matrix = np.array(confusion_matrix)
        if scale_log:
            confusion_matrix = np.log(np.ones(confusion_matrix.shape) + confusion_matrix)
        confusion_matrix = np.transpose(confusion_matrix)
        
        fig,ax=plt.subplots(figsize=(7,5))
        sns.heatmap(confusion_matrix,cmap='terrain',ax=ax)
        plt.axis('off')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)
        plt.show()
    
    def plotPrevNewAccTrendComparison(self,accuracies,labels):
        markers = ['^','o']
        colors = ['green','darkblue']

        _,ax = plt.subplots(figsize=(12,8))
        for index,(acc,label) in enumerate(zip(accuracies,labels)):
            ax.plot(np.arange(10,110, 10),acc,marker=markers[int(index/2)],color=colors[int(index/2)],label=label)

        plt.ylabel('Accuracy')
        plt.xlabel('n_classes')
        plt.title('Top-1 accuracy')
        major_ticks = np.arange(0, 1.1, 0.1)
        minor_ticks = np.arange(0, 1, 0.02)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(10,110,10))
        ax.set_xlim(xmin=9,xmax=101)
        ax.set_ylim(ymin=0.35,ymax=1.0)
        
        plt.legend()
        ax.grid(axis='y')

    def dotPlot(self,accuracies,labels,limit_min=True,colors_list=None):
        markers = ['D','o','x','+','^','*','v']

        colors = ['green','darkblue','orange','grey','black','purple','aqua']
        if colors_list != None:
            colors =colors_list

        _,ax = plt.subplots(figsize=(12,8))
        for index,(acc,label) in enumerate(zip(accuracies,labels)):
            markerline, stemlines, baseline = plt.stem(
                    np.arange(1,11), acc, linefmt='grey',use_line_collection=True, markerfmt=markers[index])
            markerline.set_markerfacecolor('none')


        plt.ylabel('Mean prediction score')
        plt.xlabel('Incremental step')

        major_ticks = np.arange(8, 18, 1)
        minor_ticks = np.arange(8, 18, 0.2)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(1,11))
        if not limit_min:
            ax.set_xlim(xmin=9,xmax=101)
            ax.set_ylim(ymin=0.0,ymax=1.0)
        else:
            ax.set_xlim(xmin=0.9,xmax=10.1)
            ax.set_ylim(ymin=6,ymax=20)
        
        plt.legend()
        ax.grid(axis='y')

    def plotAccTrendComparison(self,accuracies,labels,limit_min=True):
        markers = ['^','o','x','+','D','*','v']
        colors = ['green','darkblue','orange','grey','black','purple','aqua']

        _,ax = plt.subplots(figsize=(12,8))
        for index,(acc,label) in enumerate(zip(accuracies,labels)):
            ax.plot(np.arange(10,110, 10),acc,marker=markers[index],color=colors[index],label=label)


        plt.ylabel('Accuracy')
        plt.xlabel('n_classes')
        plt.title('Top-1 accuracy')
        major_ticks = np.arange(0, 1.1, 0.1)
        minor_ticks = np.arange(0, 1, 0.02)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_xticks(np.arange(10,110,10))
        if not limit_min:
            ax.set_xlim(xmin=9,xmax=101)
            ax.set_ylim(ymin=0.0,ymax=1.0)
        else:
            ax.set_xlim(xmin=9,xmax=101)
            ax.set_ylim(ymin=0.35,ymax=1.0)
        
        plt.legend()
        ax.grid(axis='y')
    
    def scatter_images(self,x, colors, human_readable_label):

        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 2.5})
        RS = 123
        name = np.unique(colors)
        # choose a color palette with seaborn.
        num_classes = len(np.unique(colors))
        palette = np.array(sns.color_palette("deep", num_classes))

        # create a scatter plot.
        f = plt.figure(figsize=(15, 10))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.grid()

        ax.axis('off')
        ax.axis('tight')


        # add the labels for each digit corresponding to the label
        txts = []

        for i in range(num_classes):

            # Position of each label at median of data points.

            xtext, ytext = np.median(x[colors == i, :], axis=0) + 1
            txt = ax.text(xtext, ytext, human_readable_label[i], fontsize=12)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
        plt.savefig(f"tnse_{len(name)}.png") #Store the pic locally
        plt.show()
        return f, ax, sc, txts
    
    def create_tsne(self,net,exemplars,training_set, human_readable_label):
        """
        The function plots the t-sne representation for the exemplar set
        Ex.
            human_readable_label = cifar100.human_readable_label
            create_tsne(icarl, human_readable_label)
        Params:
        net: the model chosen
        human_readable_label: the names of the label assigned to each image
        Return:
        t-sne representation of image in 2 dimensions
        """
        with torch.no_grad():
            net.eval()
            for i,exemplar_class_set in enumerate(exemplars.exemplar_set):
                dim = len(exemplar_class_set)
                fts_exemplar = []
                if i == 0:
                    all_images = []
                class_images = exemplars.get_class_images(training_set,exemplar_class_set) # recupero le immagini degli exemplars attraverso gli indici precedentemente selezionati
                for exemplar in  class_images:
                    ft_map = net.feature_extractor(exemplar.to("cuda").unsqueeze(0)).squeeze().cpu()
                    fts_exemplar.append(ft_map)
                fts_exemplar = torch.stack(fts_exemplar)

                if i == 0:
                    all_images = fts_exemplar
                    all_labels = np.full((dim), i)
                    print(all_labels)
                else:
                    all_images = torch.cat((all_images, fts_exemplar), 0)
                    all_labels = np.concatenate((all_labels, np.full((dim), i)))


            #Now I Have all_images and all_labels, I can start the reduce phase
            fashion_tsne = TSNE().fit_transform(all_images.cpu().detach().numpy())
            #Plot
            f, ax, sc, txts = self.scatter_images(fashion_tsne, all_labels, human_readable_label)
            return f, ax, sc, txts

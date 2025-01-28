# Author: tao88 Tao
# Date: 2023-10-02
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

#%% Generate plots and histograms from test results
def plot_and_hist(config_test,
                  save_path,
                  dist_correct_imgs=None, dist_corrupted_imgs=None,
                  dist_correct_imgs_for=None, dist_corrupted_imgs_for=None,
                  dist_correct_imgs_back=None, dist_corrupted_imgs_back=None,
                  dist_correct_imgs_both=None, dist_corrupted_imgs_both=None,
                  dist_batch_allperms_mean=None):
    if not config_test.puzzle_solving:
        # for paper:
        dist_correct_imgs_for, dist_corrupted_imgs_for = np.array(dist_correct_imgs_for.cpu()), np.array(dist_corrupted_imgs_for.cpu()) # both (#, 4)
        dist_correct_imgs_back, dist_corrupted_imgs_back = np.array(dist_correct_imgs_back.cpu()), np.array(dist_corrupted_imgs_back.cpu()) # both (#, 4)
        dist_correct_imgs_both, dist_corrupted_imgs_both = np.array(dist_correct_imgs_both.cpu()), np.array(dist_corrupted_imgs_both.cpu()) # both (#, 5)
        # print data
        print("Printing forward_correct_mean:")
        for item in np.mean(dist_correct_imgs_for, axis=0): print(item)
        print("Printing forward_correct_25th:")
        for item in np.percentile(dist_correct_imgs_for, 25, axis=0): print(item)
        print("Printing forward_correct_75th:")
        for item in np.percentile(dist_correct_imgs_for, 75, axis=0): print(item)
        print("Printing forward_corrupted_mean:")
        for item in np.mean(dist_corrupted_imgs_for, axis=0): print(item)
        print("Printing forward_corrupted_25th:")
        for item in np.percentile(dist_corrupted_imgs_for, 25, axis=0): print(item)
        print("Printing forward_corrupted_75th:")
        for item in np.percentile(dist_corrupted_imgs_for, 75, axis=0): print(item)
        print("Printing backward_correct_mean:")
        for item in np.mean(dist_correct_imgs_back, axis=0): print(item)
        print("Printing backward_correct_25th:")
        for item in np.percentile(dist_correct_imgs_back, 25, axis=0): print(item)
        print("Printing backward_correct_75th:")
        for item in np.percentile(dist_correct_imgs_back, 75, axis=0): print(item)
        print("Printing backward_corrupted_mean:")
        for item in np.mean(dist_corrupted_imgs_back, axis=0): print(item)
        print("Printing backward_corrupted_25th:")
        for item in np.percentile(dist_corrupted_imgs_back, 25, axis=0): print(item)
        print("Printing backward_corrupted_75th:")
        for item in np.percentile(dist_corrupted_imgs_back, 75, axis=0): print(item)
        print("Printing both_correct_mean:")
        for item in np.mean(dist_correct_imgs_both, axis=0): print(item)
        print("Printing both_correct_25th:")
        for item in np.percentile(dist_correct_imgs_both, 25, axis=0): print(item)
        print("Printing both_correct_75th:")
        for item in np.percentile(dist_correct_imgs_both, 75, axis=0): print(item)
        print("Printing both_corrupted_mean:")
        for item in np.mean(dist_corrupted_imgs_both, axis=0): print(item)
        print("Printing both_corrupted_25th:")
        for item in np.percentile(dist_corrupted_imgs_both, 25, axis=0): print(item)
        print("Printing both_corrupted_75th:")
        for item in np.percentile(dist_corrupted_imgs_both, 75, axis=0): print(item)
        
        # 5.4.2023 - tao88: make plots of distribution of distance over glimpes, for correct and corrupt images
        plt.figure()
        plt.plot(np.arange(2, 6), np.mean(dist_correct_imgs_for, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.max(dist_correct_imgs_for, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.min(dist_correct_imgs_for, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.mean(dist_corrupted_imgs_for, axis=0), 'r')
        plt.plot(np.arange(2, 6), np.max(dist_corrupted_imgs_for, axis=0), 'r')
        plt.plot(np.arange(2, 6), np.min(dist_corrupted_imgs_for, axis=0), 'r')
        plt.ylim([0, 2.5])
        plt.savefig(os.path.join(save_path, 'spread_together_forward.png'))
        plt.close()

        plt.figure()
        plt.plot(np.arange(2, 6), np.mean(dist_correct_imgs_back, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.max(dist_correct_imgs_back, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.min(dist_correct_imgs_back, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.mean(dist_corrupted_imgs_back, axis=0), 'r')
        plt.plot(np.arange(2, 6), np.max(dist_corrupted_imgs_back, axis=0), 'r')
        plt.plot(np.arange(2, 6), np.min(dist_corrupted_imgs_back, axis=0), 'r')
        plt.ylim([0, 2.5])
        plt.savefig(os.path.join(save_path, 'spread_together_backward.png'))
        plt.close()

        plt.figure()
        plt.plot(np.arange(2, 6), np.mean(dist_correct_imgs_for, axis=0) + np.mean(dist_correct_imgs_for, axis=0), 'b')
        plt.plot(np.arange(2, 6), np.mean(dist_corrupted_imgs_for, axis=0) + np.mean(dist_corrupted_imgs_back, axis=0), 'r')
        plt.ylim([0, 2.5])
        plt.savefig(os.path.join(save_path, 'spread_together_both.png'))
        plt.close()
        
        # Determine threshold and make histogram
        dist_correct_imgs, dist_corrupted_imgs = np.array(dist_correct_imgs.cpu()), np.array(dist_corrupted_imgs.cpu())
        dist_correct_mean, dist_corrupted_mean = dist_correct_imgs.sum(1), dist_corrupted_imgs.sum(1)
        hist_correct, bin_edges_correct = np.histogram(dist_correct_mean, bins=np.arange(0, 6.51, 0.05))
        hist_corrupted, bin_edges_corrupted = np.histogram(dist_corrupted_mean, bins=np.arange(0, 6.51, 0.05))
        print("Number of test samples taken into account in hist: correct: {}, corrupted: {}".format(hist_correct.sum(), hist_corrupted.sum()))
        print("Correct sample mean distance range: {} to {}".format(dist_correct_mean.min(), dist_correct_mean.max()))
        print("Corrupted sample mean distance range: {} to {}".format(dist_corrupted_mean.min(), dist_corrupted_mean.max()))

        # Determine threshold
        cutoff_index = np.where(hist_corrupted>hist_correct)[0][0]
        first_occ_threshold = bin_edges_correct[cutoff_index] # first occurence
        print("First occurence where there're more corrupted samples than correct samples: %.2f" % (first_occ_threshold))
        for cutoff_index in range(0, bin_edges_correct.shape[0]):
            threshold = bin_edges_correct[cutoff_index] # first occurence
            num_fp, num_fn = hist_correct[cutoff_index:].sum(), hist_corrupted[:cutoff_index].sum()
            num_tp, num_tn = hist_corrupted[cutoff_index:].sum(), hist_correct[:cutoff_index].sum()
            test_acc = (num_tn + num_tp) / (num_fn + num_fp + num_tn + num_tp) * 100
            det_acc = (num_tp) / (num_fn + num_tp) * 100
            print("Threshold: %.2f" %(threshold), "test accuracy is %.3f%%, det accuracy is %.3f%%, fn %d, fp %d, tn %d, tp %d" % (test_acc, det_acc, num_fn, num_fp, num_tn, num_tp))
        
        # Use plt for visualization
        plt.figure()
        plt.hist(dist_correct_mean, bins=np.arange(0, 6.55, 0.05), alpha=0.7, color='b', label='correct')
        plt.hist(dist_corrupted_mean, bins=np.arange(0, 6.55, 0.05), alpha=0.7, color='r', label='corrupted')
        plt.xticks(np.arange(0, 6.5, 0.5))
        plt.yticks(np.arange(0, 700, 100))
        # plt.title('LSTM Prediction Results')
        plt.xlabel('Total residual', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.legend(["correct", "corrupted"], fontsize="18", loc ="upper right")
        plt.savefig(os.path.join(save_path, 'hist.png'))
        plt.close()
        print("Histogram on test plotted and saved")
        print("Finished testing.\n")
    else:
        dist_batch_allperms_mean = np.array(dist_batch_allperms_mean.cpu()) # (num_samples, num_permute+1)
        # if entry 0 < all other entries, correct prediction!
        num_correct = (dist_batch_allperms_mean[:,0]<dist_batch_allperms_mean[:,1:].min(axis=1)).sum()
        num_samples = dist_batch_allperms_mean.shape[0]
        test_acc = (num_correct / num_samples) *100
        print("Number of samples when the correct puzzle is picked: %d" % (num_correct))
        print("Total count of samples in the testset: %d" % (num_samples))
        print("The test accuracy is %.3f%%" % (test_acc))

        # Histogram for visualizing distribution of correct vs. fake puzzles
        plt.figure()
        plt.hist(dist_batch_allperms_mean[:,0], bins=np.arange(0, 1.6, 0.05), alpha=0.7, color='b', label='correct')
        plt.hist(dist_batch_allperms_mean[:,1:].flatten(), bins=np.arange(0, 1.6, 0.05), alpha=0.7, color='r', label='corrupted')
        plt.xticks(np.arange(0, 1.6, 0.1))
        plt.yticks(np.arange(0, 2100, 100))
        plt.title('LSTM Prediction Results')
        plt.xlabel('Average dist. over 5 patches')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(save_path, 'hist_puzzle_solving.png'))
        plt.close()
        print("Histogram on test plotted and saved")
        print("Finished testing.\n")


#%% Generate averaged mask for each episode in the sequence
def compute_average_mask(save_path, 
                         sum_masks_all_batches,
                         sum_semantics_all_batches,
                         sum_pred_semantics_forward_all_batches,
                         sum_pred_semantics_backward_all_batches,
                         count_samples,
                         cmap):
    # sum_masks_all_batches has shape (5, 64, 64)
    avg_masks_all_batches = torch.round(sum_masks_all_batches / count_samples)
    # torch.save(avg_masks_all_batches, './LSTM_test_results/celeba_avg_masks_testset.pt')

    avg_semantics_all_batches = sum_semantics_all_batches / count_samples
    # torch.save(avg_semantics_all_batches, './LSTM_test_results/celeba_avg_semantics_testset.pt')

    avg_pred_semantics_forward_all_batches = sum_pred_semantics_forward_all_batches / count_samples
    avg_pred_semantics_backward_all_batches = sum_pred_semantics_backward_all_batches / count_samples
    for patch_id in range(5):
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        fig.add_axes(ax)
        psm = ax.imshow(avg_masks_all_batches[patch_id].squeeze().cpu().numpy(),
                        interpolation='nearest',
                        cmap=cmap,
                        vmin=0,
                        vmax=7)
        cbar = fig.colorbar(psm, ticks=[0, 1, 2, 3, 4, 5, 6])
        # cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
        plt.savefig(os.path.join(save_path, 'avgpatch_{}_mask.png'.format(patch_id)))
        plt.close(fig)
    print("Average semantics:", avg_semantics_all_batches) #(5, 5)
    print("Average predicted semantics in forward sequence:", avg_pred_semantics_forward_all_batches) #(4, 5)
    print("Average predicted semantics in backward sequence:", avg_pred_semantics_backward_all_batches)
    print("Difference between average predicted semantics and average true semantics in forward sequence:", torch.norm((avg_pred_semantics_forward_all_batches - avg_semantics_all_batches[1:]), dim=1, keepdim=False))
    print("Difference between average predicted semantics and average true semantics in backward sequence:", torch.norm((avg_pred_semantics_backward_all_batches - avg_semantics_all_batches[:-1]), dim=1, keepdim=False))


def calculate_iou(pred_mask, true_mask, num_classes):
    iou = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        pred_class_mask = (pred_mask == class_idx).float()
        true_class_mask = (true_mask == class_idx).float()
        intersection = torch.sum(pred_class_mask * true_class_mask)
        union = torch.sum(pred_class_mask) + torch.sum(true_class_mask) - intersection
        # avoid division by zero
        iou[class_idx] = (intersection + 1e-6) / (union + 1e-6)
    return iou
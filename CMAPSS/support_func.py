import numpy as np
import matplotlib.pyplot as plt

def plot_results(results_list,partno):
	models = [x[0] for x in results_list]
	if partno==1:
		cv_score_train_mean = [x[1][0] for x in results_list]
		cv_score_val_mean = [x[1][1] for x in results_list]
		cv_score_test_mean = [x[1][2] for x in results_list]
		# Getting accuracies from conf matrices
		pct_score_train_mean = [np.round(np.trace(x[3][0])/np.sum(x[3][0])*100,2) for x in results_list]
		pct_score_val_mean = [np.round(np.trace(x[3][1])/np.sum(x[3][1])*100,2) for x in results_list]
		pct_score_test_mean = [np.round(np.trace(x[3][2])/np.sum(x[3][2])*100,2) for x in results_list]
		overestimated_train = [np.triu(x[3][0],k=1).sum() for x in results_list]
		overestimated_val = [np.triu(x[3][1],k=1).sum() for x in results_list]
		overestimated_test = [np.triu(x[3][2],k=1).sum() for x in results_list]
		underestimated_train = [np.tril(x[3][0],k=-1).sum() for x in results_list]
		underestimated_val = [np.tril(x[3][1],k=-1).sum() for x in results_list]
		underestimated_test = [np.tril(x[3][2],k=-1).sum() for x in results_list]
		plot_rows = 2
		plot_cols = 2
		plt.figure(figsize=(15,5))
		plt.subplot(plot_rows,plot_cols,1)
		plt.title('CV scores')
		plt.plot(models,cv_score_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,cv_score_val_mean,label='val')
		plt.plot(models,cv_score_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,2)
		plt.title('% acc')
		plt.plot(models,pct_score_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,pct_score_val_mean,label='val')
		plt.plot(models,pct_score_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,3)
		plt.title('Overestimated TTF')
		plt.plot(models,overestimated_train,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,overestimated_val,label='val')
		plt.plot(models,overestimated_test,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,4)
		plt.title('Underestimated TTF')
		plt.plot(models,underestimated_train,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,underestimated_val,label='val')
		plt.plot(models,underestimated_test,label='test')
		plt.legend()
		return models, overestimated_train, overestimated_val, overestimated_test, underestimated_train, underestimated_val, underestimated_test, pct_score_train_mean, pct_score_val_mean, pct_score_test_mean
	if partno==2:
		f1_micro_train_mean = [x[4][0][0] for x in results_list]
		f1_micro_val_mean = [x[4][1][0] for x in results_list]
		f1_micro_test_mean = [x[4][2][0] for x in results_list]
		f1_macro_train_mean = [x[4][0][1] for x in results_list]
		f1_macro_val_mean = [x[4][1][1] for x in results_list]
		f1_macro_test_mean = [x[4][2][1] for x in results_list]
		f1_weighted_train_mean = [x[4][0][2] for x in results_list]
		f1_weighted_val_mean = [x[4][1][2] for x in results_list]
		f1_weighted_test_mean = [x[4][2][2] for x in results_list]
		accuracy_train_mean = [x[4][0][3] for x in results_list]
		accuracy_val_mean = [x[4][1][3] for x in results_list]
		accuracy_test_mean = [x[4][2][3] for x in results_list]
		plot_rows = 2
		plot_cols = 2
		plt.figure(figsize=(15,5))
		plt.subplot(plot_rows,plot_cols,1)
		plt.title('f1_micro')
		plt.plot(models,f1_micro_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,f1_micro_val_mean,label='val')
		plt.plot(models,f1_micro_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,2)
		plt.title('f1_macro')
		plt.plot(models,f1_macro_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,f1_macro_val_mean,label='val')
		plt.plot(models,f1_macro_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,3)
		plt.title('f1_weighted')
		plt.plot(models,f1_weighted_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,f1_weighted_val_mean,label='val')
		plt.plot(models,f1_weighted_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,4)
		plt.title('accuracy')
		plt.plot(models,accuracy_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,accuracy_val_mean,label='val')
		plt.plot(models,accuracy_test_mean,label='test')
		plt.legend()
	if partno==3:
		precision_micro_train_mean = [x[4][0][4] for x in results_list]
		precision_micro_val_mean = [x[4][1][4] for x in results_list]
		precision_micro_test_mean = [x[4][2][4] for x in results_list]
		precision_macro_train_mean = [x[4][0][5] for x in results_list]
		precision_macro_val_mean = [x[4][1][5] for x in results_list]
		precision_macro_test_mean = [x[4][2][5] for x in results_list]
		precision_weighted_train_mean = [x[4][0][6] for x in results_list]
		precision_weighted_val_mean = [x[4][1][6] for x in results_list]
		precision_weighted_test_mean = [x[4][2][6] for x in results_list]
		recall_micro_train_mean = [x[4][0][7] for x in results_list]
		recall_micro_val_mean = [x[4][1][7] for x in results_list]
		recall_micro_test_mean = [x[4][2][7] for x in results_list]
		recall_macro_train_mean = [x[4][0][8] for x in results_list]
		recall_macro_val_mean = [x[4][1][8] for x in results_list]
		recall_macro_test_mean = [x[4][2][8] for x in results_list]
		recall_weighted_train_mean = [x[4][0][9] for x in results_list]
		recall_weighted_val_mean = [x[4][1][9] for x in results_list]
		recall_weighted_test_mean = [x[4][2][9] for x in results_list]
		plot_rows = 2
		plot_cols = 3
		plt.figure(figsize=(15,5))
		plt.subplot(plot_rows,plot_cols,1)
		plt.title('precision_micro')
		plt.plot(models,precision_micro_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,precision_micro_val_mean,label='val')
		plt.plot(models,precision_micro_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,2)
		plt.title('precision_macro')
		plt.plot(models,precision_macro_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,precision_macro_val_mean,label='val')
		plt.plot(models,precision_macro_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,3)
		plt.title('precision_weighted')
		plt.plot(models,precision_weighted_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,precision_weighted_val_mean,label='val')
		plt.plot(models,precision_weighted_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,4)
		plt.title('recall_micro')
		plt.plot(models,recall_micro_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,recall_micro_val_mean,label='val')
		plt.plot(models,recall_micro_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,5)
		plt.title('recall_macro')
		plt.plot(models,recall_macro_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,recall_macro_val_mean,label='val')
		plt.plot(models,recall_macro_test_mean,label='test')
		plt.legend()
		plt.subplot(plot_rows,plot_cols,6)
		plt.title('recall_weighted')
		plt.plot(models,recall_weighted_train_mean,label='train',linewidth=0.5,linestyle='dashed')
		plt.plot(models,recall_weighted_val_mean,label='val')
		plt.plot(models,recall_weighted_test_mean,label='test')
		plt.legend()


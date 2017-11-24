import numpy as np;
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

def load_time_series(filename):
  data = np.loadtxt(filename)
  time = data[:, 0]
  data = data[:, 1:]
  return (time, data)

def get_time_series_sequence(time_series):
  return np.reshape(time_series, np.prod(time_series.shape))

def plot_time_series(time_series, pred_time_series):
  import pdb; pdb.set_trace()
  plt.plot(time_series)
  plt.plot(pred_time_series, '--o')
  plt.grid()
  plt.show()

def show_series_info(time_series):
  time_series_seq = get_time_series_sequence(time_series)
  print "Series Mean %f, Std %f" % (np.mean(time_series_seq), np.std(time_series_seq))
  print "Montly mean: ", np.mean(time_series, axis=0)
  print "Montly std: ", np.std(time_series, axis=0)

def standard_time_series(time_series):
  averages = np.mean(time_series, axis=0)
  stds = np.std(time_series, axis=0)
  std_series = (time_series - averages)/stds
  return std_series, averages, stds

def get_regression_matrix(time_series, month, ar_order, predict_len):
  data_matrix = toeplitz(np.zeros(ar_order,), np.concatenate((np.zeros((predict_len,)), time_series[:-predict_len, month])))
  target = time_series[:, month]
  return (data_matrix, target)

def get_ar_coefitients(time_series, month, ar_order, predict_len):
  std_time_series, averages, std_devs = standard_time_series(time_series)
  data_matrix, target = get_regression_matrix(std_time_series, month, ar_order, predict_len)
  lst_sol = np.linalg.lstsq(data_matrix.T, target)
  predicted_series = np.dot(lst_sol[0], data_matrix)*std_devs[month] + averages[month]
  error =  predicted_series - time_series[:, month]
  return lst_sol[0], error, predicted_series

def main():
  show_info = False
  show_plot = True

  # Loading time series data
  (time, time_series) = load_time_series("vazao_furnas")

  # Fitting periodic auto-regressive model, batch mode
  ar_order = 5;
  predict_len = 1;
  predicted_series_ar = np.zeros(time_series.shape)
  series_error_ar = np.zeros(time_series.shape)
  for month in range(0, 12):
    print "Processing month %d" % month
    ls_sol, series_error_ar[:, month], predicted_series_ar[:, month] = get_ar_coefitients(time_series, month, ar_order, predict_len)

  print "MSE: %f" % np.mean(series_error_ar**2)

  # Fitting RedigeRegression
  rigde_order = 12;
  predicted_series_rigde = np.zeros(time_series.shape)
  series_error_rigde = np.zeros(time_series.shape)
  std_time_series, averages, std_devs = standard_time_series(time_series)
  rigde_rbf = KernelRidge(alpha=1e9)
  for month in range(0, 12):
    print "Processing month %d" % month
    data_matrix, target = get_regression_matrix(std_time_series, month, rigde_order, predict_len)
    predicted_series_rigde[:, month] = rigde_rbf.fit(data_matrix.T, target).predict(data_matrix.T)*std_devs[month] + averages[month]
    series_error_rigde[:, month] = predicted_series_rigde[:, month] - time_series[:, month]

  print "MSE: %f" % np.mean(series_error_rigde**2)

  # Fitting SVR
  svr_order = 12;
  predicted_series_rbf = np.zeros(time_series.shape)
  series_error_rbf = np.zeros(time_series.shape)
  std_time_series, averages, std_devs = standard_time_series(time_series)
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=1)
  for month in range(0, 12):
    print "Processing month %d" % month
    data_matrix, target = get_regression_matrix(std_time_series, month, svr_order, predict_len)
    predicted_series_rbf[:, month] = svr_rbf.fit(data_matrix.T, target).predict(data_matrix.T)*std_devs[month] + averages[month]
    series_error_rbf[:, month] = predicted_series_rbf[:, month] - time_series[:, month]

  print "MSE: %f" % np.mean(series_error_rbf**2)


  if(show_info):
    show_series_info(time_series)
    show_series_info(std_time_series)
  if(show_plot):
    time_series_seq = get_time_series_sequence(time_series)
    pred_time_series_seq = get_time_series_sequence(predicted_series_rbf)
    plot_time_series(time_series_seq, pred_time_series_seq)


if __name__ == '__main__':
  main()

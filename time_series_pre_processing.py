import numpy as np;
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

def load_time_series(filename):
  data = np.loadtxt(filename)
  time = data[:, 0]
  data = data[:, 1:]
  return (time, data)

def get_time_series_sequence(time_series):
  return np.reshape(time_series, np.prod(time_series.shape))

def plot_time_series(time_series):
  plt.plot(time_series)
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

def get_ar_coefitients(time_series, month, ar_order, predict_len):
  data_matrix = toeplitz(np.zeros(ar_order,), np.concatenate((np.zeros((predict_len,)), time_series[:-predict_len, month])))
  target = time_series[:, month]
  #np.dot(np.dot(np.linalg.inv(np.dot(data_matrix, data_matrix.T)), data_matrix), target)
  lst_sol = np.linalg.lstsq(data_matrix.T, target)
  mse =  np.mean((np.dot(lst_sol[0], data_matrix)-target)**2)
  return lst_sol[0], mse

def main():
  show_info = False
  show_plot = False

  # Loading time series data
  (time, time_series) = load_time_series("vazao_furnas")

  # Standarizing time series/removing montly sazonality
  std_time_series, averages, std_devs = standard_time_series(time_series)
  ls_sol, mse = get_ar_coefitients(std_time_series, 1, 7, 1)
  mse_original_series = mse*std_devs[0]
  import pdb; pdb.set_trace()
  if(show_info):
    show_series_info(time_series)
    show_series_info(std_time_series)
  if(show_plot):
    std_time_series_seq = get_time_series_sequence(std_time_series)
    plot_time_series(std_time_series_seq)


if __name__ == '__main__':
  main()

remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Kuzushiji 49 Dataset..."
  rm *.npz
  rm *.csv
  echo "Removed!"
else
  echo "Downloading Kuzushiji 49 Dataset..."
  wget "http://codh.rois.ac.jp/kmnist/dataset/k49/k49_classmap.csv" -O k49_classmap.csv
  wget "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz" -O k49-train-imgs.npz
  wget "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz" -O k49-train-labels.npz
  wget "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz" -O k49-test-imgs.npz
  wget "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz" -O k49-test-labels.npz
  echo "Download Finished!"
fi
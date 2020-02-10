remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Coil 100 Dataset..."
  rm *.zip
  rm -rf files
  echo "Removed!"
else
  echo "Downloading Coil 100 Dataset..."
  wget "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip" -O coil-100.zip
  echo "Extracting Coil 100..."
  unzip -d files -q coil-100.zip
  mv files/coil-100/*  files/
  rm -rf files/coil-100
  echo "Download Finished!"
fi
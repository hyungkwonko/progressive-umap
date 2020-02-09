remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Coil 100 Dataset..."
  rm *.zip
  rm -rf coil-100
  echo "Removed!"
else
  echo "Downloading Coil 100 Dataset..."
  wget "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip" -O coil-100.zip
  echo "Extracting Coil 100..."
  unzip -q coil-100.zip
  echo "Download Finished!"
fi
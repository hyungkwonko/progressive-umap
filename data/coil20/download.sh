remove=$1
if [ "$remove" = "remove" ]; then
  echo "Removing Coil 20 Dataset..."
  rm *.zip
  rm -rf files
  echo "Removed!"
else
  echo "Downloading Coil 20 Dataset..."
  wget "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip" -O coil-20-proc.zip
  echo "Extracting Coil 20..."
  unzip -d files -q coil-20-proc.zip
  mv files/coil-20-proc/*  files/
  rm -rf files/coil-20-proc
  echo "Download Finished!"
fi
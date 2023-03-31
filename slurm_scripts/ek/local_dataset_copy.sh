rm -rf /raid/local_scratch/txp48-wwp01/frames

cp -r /jmain02/home/J2AD001/wwp01/shared/data/epic-100/frames /raid/local_scratch/txp48-wwp01/


for fol in /raid/local_scratch/txp48-wwp01/frames/*; do
   echo $fol
   cd $fol
        for f in *.tar; do folder=${f::-4}; mkdir -p $folder; cd $folder; tar -xf ../$f . ; cd ..; done
   cd ..
done
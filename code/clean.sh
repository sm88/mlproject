#! /bin/bash
if [[ $# -ne 1 ]]; then
	echo "usage ./clean.sh filename"
	exit 1
fi

if ! [[ -e $1 ]]; then
	echo "invalid file"
	exit 1
fi

#clean
echo "cleaning..."
sed -i -e 's#[ ]*\[[ ]*##g' -e 's#[ ]*\][ ]*##g' -e 's/^[0-9]\+---\([a-zA-Z]\+\)/\1#/g' -e "s/[\",;'\(\)-]//g" -e 's/[ ]\{2,\}/ /g' -e 's/[ ]*\([.!\?]\)[ ]*/\1 /g' $1
#sed -i -e 's#[ ]*\[[ ]*##g' $1 
#sed -i -e 's#[ ]*\][ ]*##g' $1 
#sed -i -e 's/[0-9]\+---//g' $1 
#sed -i -e 's/---/,/g' $1
echo "done"


#dbus-launch bash

if test -z "$DBUS_SESSION_BUS_ADDRESS" 
then
     ## if not found, launch a new one
     eval `dbus-launch --sh-syntax`
     echo "D-Bus per-session daemon address is: $DBUS_SESSION_BUS_ADDRESS"
fi

echo -e "\e[33mHi, please login at rob_file with your JR credentials (Domain: jr1)\e[0m"
gvfs-mount smb://rzjna01/rob_file/

gvfs-copy -i -p smb://rzjna01/rob_file/staff/wol/oaho/data/oaho_synth_train.tfrecord .
gvfs-copy -i -p smb://rzjna01/rob_file/staff/wol/oaho/data/oaho_synth_val.tfrecord .
gvfs-copy -i -p smb://rzjna01/rob_file/staff/wol/oaho/data/oaho_synth_test.tfrecord .

#gvfs-mount -u smb://rzjna01/rob_file/

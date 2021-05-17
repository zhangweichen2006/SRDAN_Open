PSWD=asdf2345
git add .
git commit -m sync_$1
# git push myrepo $1
git checkout master
# git stash
git pull myrepo master
git checkout $1
git rebase master

#git pull my
 #-u myrepo master
#rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti3ssd/
# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/
# sshpass -p $PSWD rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron wzha8158@10.66.30.57:~/Dropbox/CVPR2021_LidarPerceptron/Ti5/
# sshpass -p $PSWD rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron wzha8158@10.66.30.66:~/Dropbox/CVPR2021_LidarPerceptron/Ti3ssd/

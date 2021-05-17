PSWD=asdf2345
git checkout master
git pull origin master
git checkout $1
git rebase master
# git add .
# git commit -m "$2 rebase master"
git push origin $1
git checkout master
git merge $1 # cherry pick!
git pull origin master
git add .
git commit -m "$2"
# git checkout -b 'new'
git push origin master
git checkout $1

#git pull my
 #-u myrepo master
#rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti3ssd/
# rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_NewDet3D/Home/Det3D ~/Dropbox/CVPR2021_NewDet3D/Ti5/
# sshpass -p $PSWD rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron wzha8158@10.66.30.57:~/Dropbox/CVPR2021_LidarPerceptron/Ti5/
# sshpass -p $PSWD rsync -avzm --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ~/Dropbox/CVPR2021_LidarPerceptron/Home/OpenLidarPerceptron wzha8158@10.66.30.66:~/Dropbox/CVPR2021_LidarPerceptron/Ti3ssd/

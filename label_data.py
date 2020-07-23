f = open("/home/liang/PycharmProjects/time-series-classification/data/target_cp", "a")

num_folers = 37
num_files  = 100
for i in range(1, num_folers+1):
    for j in range(1, num_files+1):
        f.write(str((i-1)*100+j)+' '+str(i)+'\n')


f.close()





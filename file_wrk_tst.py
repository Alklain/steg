f = open('input.txt')
mark=[]
for line in f:
    if len(line)>1:
        mark.append(int(line))
f.close()
print(mark)

#Out_dat = [1, 4, 5, 6]
#print(Out_dat)

#f = open('output.txt', 'w')
#for i in range(len(Out_dat)):
##    f.write(str(Out_dat[i]) + '\n')
#f.close()
#print('done')
import os,re

PATH = "./Paper4Data_raw/"
try:
    shutil.rmtree('Paper4Data_treated')
except:
    pass
finally:
    os.mkdir('Paper4Data_treated')


SAVEPATH = "./Paper4Data_treated/"

List = os.listdir(PATH)

# ONLY KEEP CHARACTERS AND PUNCTUATE
for file in List:
    fp = open(SAVEPATH+file,'w')
    lines = open(PATH+file).readlines()
    for line in lines:
        newline = ''
        for s in line:
            if s.isalpha() or s is ' ' or s is '\n' or s in ',.()-':
                newline += s
        if len(newline) < 10:
            continue
        e = re.compile('\(.*?\)')
        newline = e.sub('', newline)
        newline += ' '
        fp.write(newline.replace('\n','')+' ')
    fp.close()

# REOGNISE TEN PAPER INTO ONE TXT
for ii in range(10):
    fp = open(SAVEPATH+str(ii)+'.txt','w')
    for jj in [1,2,3,4,5,6,7,8,9,10]:
        try:
            lines = open(SAVEPATH+str(ii)+'.'+str(jj)+'.pdf.txt').readlines()
        except :
            pass
        else:
            for line in lines:
                fp.write(line.replace('\n','')+' ')
    fp.close()

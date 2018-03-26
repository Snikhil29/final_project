import csv

tpos=0
tneg=0
fpos=0
fneg=0

def Accuracy(truepos, trueneg, falsepos, falseneg):
    a = (truepos + trueneg) / float(truepos + trueneg + falsepos + falseneg)
    return (a)

def Precision(truepos, trueneg, falsepos, falseneg):
    a = (truepos) / float(truepos + falsepos)
    return (a)

def ReCall(truepos, trueneg, falsepos, falseneg):
    a = (truepos) / float(truepos + falseneg)
    return (a)

def F_Measure(truepos, trueneg, falsepos, falseneg):
    p = Precision(truepos, trueneg, falsepos, falseneg)
    r = ReCall(truepos, trueneg, falsepos, falseneg)
    i = 2 * ( (p*r) / float(p+r) )
    return (i)
print "bi gram"
print "max features 1250"

with open("output1\\resultF3.csv") as myfile:
    reader = csv.reader(myfile,  quoting=3)
    for val in reader:
        if(val[0] == '1' and val[2] == '1'):
            tpos = tpos + 1
        elif(val[0] == '1' and val[2] == '0'):
            fpos = fpos + 1
        elif(val[0] == '0' and val[2] == '0'):
            tneg = tneg + 1
        else:
            fneg = fneg + 1

print "RF UNI........................................."
print(tpos)
print(tneg)
print(fpos)
print(fneg)
print('Accuracy percentage:',Accuracy(tpos,tneg,fpos,fneg-1)*100)
print('Precision percentage',Precision(tpos,tneg,fpos,fneg-1)*100)
print('ReCall percentage',ReCall(tpos,tneg,fpos,fneg-1)*100)
print('F_Measure percentage',F_Measure(tpos,tneg,fpos,fneg-1)*100)







tpos=0
tneg=0
fpos=0
fneg=0
with open("output1\\resultAB3.csv") as myfile:
    reader = csv.reader(myfile,  quoting=3)
    for val in reader:
        if(val[0] == '1' and val[2] == '1'):
            tpos = tpos + 1
        elif(val[0] == '1' and val[2] == '0'):
            fpos = fpos + 1
        elif(val[0] == '0' and val[2] == '0'):
            tneg = tneg + 1
        else:
            fneg = fneg + 1

print "AB UNI........................................."
print(tpos)
print(tneg)
print(fpos)
print(fneg)
print('Accuracy percentage:',Accuracy(tpos,tneg,fpos,fneg-1)*100)
print('Precision percentage',Precision(tpos,tneg,fpos,fneg-1)*100)
print('ReCall percentage',ReCall(tpos,tneg,fpos,fneg-1)*100)
print('F_Measure percentage',F_Measure(tpos,tneg,fpos,fneg-1)*100)




tpos=0
tneg=0
fpos=0
fneg=0
with open("output1\\resultGB3.csv") as myfile:
    reader = csv.reader(myfile,  quoting=3)
    for val in reader:
        if(val[0] == '1' and val[2] == '1'):
            tpos = tpos + 1
        elif(val[0] == '1' and val[2] == '0'):
            fpos = fpos + 1
        elif(val[0] == '0' and val[2] == '0'):
            tneg = tneg + 1
        else:
            fneg = fneg + 1

print "GB UNI........................................."
print(tpos)
print(tneg)
print(fpos)
print(fneg)
print('Accuracy percentage:',Accuracy(tpos,tneg,fpos,fneg-1)*100)
print('Precision percentage',Precision(tpos,tneg,fpos,fneg-1)*100)
print('ReCall percentage',ReCall(tpos,tneg,fpos,fneg-1)*100)
print('F_Measure percentage',F_Measure(tpos,tneg,fpos,fneg-1)*100)




tpos=0
tneg=0
fpos=0
fneg=0
with open("output1\\resultBAG3.csv") as myfile:
    reader = csv.reader(myfile,  quoting=3)
    for val in reader:
        if(val[0] == '1' and val[2] == '1'):
            tpos = tpos + 1
        elif(val[0] == '1' and val[2] == '0'):
            fpos = fpos + 1
        elif(val[0] == '0' and val[2] == '0'):
            tneg = tneg + 1
        else:
            fneg = fneg + 1

print "BAG UNI........................................."
print(tpos)
print(tneg)
print(fpos)
print(fneg)
print('Accuracy percentage:',Accuracy(tpos,tneg,fpos,fneg-1)*100)
print('Precision percentage',Precision(tpos,tneg,fpos,fneg-1)*100)
print('ReCall percentage',ReCall(tpos,tneg,fpos,fneg-1)*100)
print('F_Measure percentage',F_Measure(tpos,tneg,fpos,fneg-1)*100)





tpos=0
tneg=0
tpos=0
tneg=0
fpos=0
fneg=0
with open("output1\\resultET3.csv") as myfile:
    reader = csv.reader(myfile,  quoting=3)
    for val in reader:
        if(val[0] == '1' and val[2] == '1'):
            tpos = tpos + 1
        elif(val[0] == '1' and val[2] == '0'):
            fpos = fpos + 1
        elif(val[0] == '0' and val[2] == '0'):
            tneg = tneg + 1
        else:
            fneg = fneg + 1

print "ET UNI........................................."
print(tpos)
print(tneg)
print(fpos)
print(fneg)
print('Accuracy percentage:',Accuracy(tpos,tneg,fpos,fneg-1)*100)
print('Precision percentage',Precision(tpos,tneg,fpos,fneg-1)*100)
print('ReCall percentage',ReCall(tpos,tneg,fpos,fneg-1)*100)
print('F_Measure percentage',F_Measure(tpos,tneg,fpos,fneg-1)*100)

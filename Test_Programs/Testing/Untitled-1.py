from itertools import count


count=0
while True:
    count+=1
    number=792*count
    if len(str(number))!=5:
        next
    if len(str(number))>5:
        break
    elif "1" in str(number) and "2" in str(number) and "3" in str(number) and "4" in str(number):
        print(number)


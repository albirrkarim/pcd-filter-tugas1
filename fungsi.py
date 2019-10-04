import numpy as np

def countQuality(imgA,imgB):
    
    height,width = imgA.shape
  
    sum=0.0
    for i in range(0,height):
        for j in range(0,width):
            a=0
            b=0
            a=imgA[i][j]
            b=imgB[i][j]
            
            if(a!=b):
                print("BEDAAAAAA")

            sum+=(a-b)**2
            # print(a)
            # print(b)

    # print(sum)
    sum=sum/(height*width)

    return sum

def ftAritmatik(image):
    img = image
    height,width = image.shape
    height=height-1
    width=width-1
    for i in range(1,height):
        for j in range(1,width):
            sum=0
            sum += image[i-1][j-1] 
            sum += image[i-1][j] 
            sum += image[i-1][j+1]
            sum += image[i][j-1] 
            sum += image[i][j] 
            sum += image[i][j+1]
            sum += image[i+1][j-1]
            sum += image[i+1][j] 
            sum += image[i+1][j+1]
            sum = sum/9
            sum = int(sum)
  
            img[i][j]=sum
    return img


def ftMedian(image):
    img = image
    height,width = image.shape
    height=height-1
    width=width-1
    
    a=0
    for i in range(1,height):
        for j in range(1,width):
            arr=[]
            a= image[i-1][j-1]
            arr.append(a)

            a= image[i-1][j]
            arr.append(a)

            a= image[i-1][j+1]
            arr.append(a)

            a= image[i][j-1]
            arr.append(a)

            a= image[i][j]
            arr.append(a)

            a= image[i][j+1]
            arr.append(a)

            a= image[i+1][j-1]
            arr.append(a)

            a= image[i+1][j]
            arr.append(a)

            a= image[i+1][j+1]
            arr.append(a)
            arr=QuickSort(arr)
            leng=len(arr)-1
      
            img[i][j]=arr[int(leng/2)]

    return img

def ftAlphaTrimmedMean(image):
    img = image
    height,width = image.shape
    height=height-1
    width=width-1
    
    a=0
    for i in range(1,height):
        for j in range(1,width):
            arr=[]
            a= image[i-1][j-1]
            arr.append(a)

            a= image[i-1][j]
            arr.append(a)

            a= image[i-1][j+1]
            arr.append(a)

            a= image[i][j-1]
            arr.append(a)

            a= image[i][j]
            arr.append(a)

            a= image[i][j+1]
            arr.append(a)

            a= image[i+1][j-1]
            arr.append(a)

            a= image[i+1][j]
            arr.append(a)

            a= image[i+1][j+1]
            arr.append(a)
            arr=QuickSort(arr)
            leng=len(arr)-1

            tengah=int(leng/2)
            a=0
            a+=arr[tengah-2]
            a+=arr[tengah-1]
            a+=arr[tengah]
            a+=arr[tengah+1]
            a+=arr[tengah+2]

            a=int(a/5)

            img[i][j]=a

    return img


def QuickSort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)

        return QuickSort(less)+equal+QuickSort(greater)  
    else:  
        return array










def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return 
        

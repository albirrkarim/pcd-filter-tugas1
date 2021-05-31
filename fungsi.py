import numpy as np


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def ftAritmatik(image):
    img = image.copy()
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
    img = image.copy()
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
    #deep copy
    img = image.copy()

    # Get image height and width
    height,width = image.shape
    
    height=height-1
    width=width-1
    
    a=0

    # loop through
    for i in range(1,height):
        for j in range(1,width):
            arr=[]

            # get pixel value and append it to array
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

            # Sorting
            arr=QuickSort(arr)
            leng=len(arr)-1

            # get minddle index
            middleIndex=int(leng/2)

            total=0

            total+=arr[middleIndex-2]
            total+=arr[middleIndex-1]
            total+=arr[middleIndex]
            total+=arr[middleIndex+1]
            total+=arr[middleIndex+2]

            total=int(total/5)

            # set pixel value back to image
            img[i][j]=total

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
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[tuple(coords)] = 0
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
        


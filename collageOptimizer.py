import cv2
import numpy as np
import math

def create_population(n):

    population = []

    for i in range(n):
        population.append(create_individual())

    return np.array(population)

def create_individual():

    ind = []

    for i in range(len(images)):
        ind.append(create_individual_part())

    return ind

def create_individual_part():

    ind = []

    ind.append(np.random.uniform(0,backgroundArea))        # y
    ind.append(np.random.uniform(0, backgroundArea))       # x
    ind.append(np.random.uniform(0.5,1))                    # escalaY
    ind.append(np.random.uniform(0.5,1))                   # escalaX
    ind.append(np.random.uniform(-math.pi/4,math.pi/4))                # rotación

    return ind

def create_image_from_individual(ind):

    background = np.zeros([backgroundArea, backgroundArea, 3], dtype=np.uint8)
    background.fill(255)

    for k in range(len(images)):
        for i in range(images[k].shape[0]):
            for j in range(images[k].shape[1]):
                x = float(j)
                y = float(i)
                x2 = x * math.cos(ind[k][4]) - y * math.sin(ind[k][4])
                y2 = x * math.sin(ind[k][4]) + y * math.cos(ind[k][4])
                x = x2 + ind[k][1]
                y = y2 + ind[k][0]
                x = int(x * ind[k][3])
                y = int(y * ind[k][2])
                if x > 0 and y > 0 and x < background.shape[1] and y < background.shape[0]:
                    background[y][x] = images[k][i][j]

    return background

def create_matrix_from_individual(ind):

    background = np.zeros([backgroundArea, backgroundArea], dtype=np.uint8)

    for k in range(len(images)):
        for i in range(images[k].shape[0]):
            for j in range(images[k].shape[1]):
                x = float(j)
                y = float(i)
                x2 = x * math.cos(ind[k][4]) - y * math.sin(ind[k][4])
                y2 = x * math.sin(ind[k][4]) + y * math.cos(ind[k][4])
                x = x2 + ind[k][1]
                y = y2 + ind[k][0]
                x = int(x * ind[k][3])
                y = int(y * ind[k][2])
                if x > 0 and y > 0 and x < background.shape[1] and y < background.shape[0]:
                    background[y][x] = k+1

    return background

def thresh_related_fitness(numberMatrix, ind):

    background = np.zeros([backgroundArea, backgroundArea, 3], dtype=np.uint8)
    penalizationCount = 0
    cont = 0

    for k in range(len(thresh)):
        aux = 0
        total = 0
        for i in range(thresh[k].shape[0]):
            for j in range(thresh[k].shape[1]):
                x = float(j)
                y = float(i)
                x2 = x * math.cos(ind[k][4]) - y * math.sin(ind[k][4])
                y2 = x * math.sin(ind[k][4]) + y * math.cos(ind[k][4])
                x = x2 + ind[k][1]
                y = y2 + ind[k][0]
                x = int(x * ind[k][3])
                y = int(y * ind[k][2])
                if x > 0 and y > 0 and x < background.shape[1] and y < background.shape[0]:
                    background[y][x] = thresh[k][i][j]
                    if np.array_equal(thresh[k][i][j],[255,255,255]):
                        total += 1
                        if numberMatrix[y][x] == k+1:
                            aux += 1
        if aux < total*minThreshAppearance:
            penalizationCount += 1
        if aux >= total*threshAppearance:
            cont += aux

    if penalizationCount > 0: return background, 0, penalizationCount
    return background, cont/len(thresh), penalizationCount

def get_fitness(population):

    fitness = []

    for ind in population:
        fitness.append(get_fitness_individual(ind))

    return np.array(fitness)

def get_fitness_individual(ind):

    background = create_image_from_individual(ind)
    numberMatrix = create_matrix_from_individual(ind)
    threshMap, threshAverage, penalizationCount = thresh_related_fitness(numberMatrix, ind)

    # higher fitness is better --> means there are no white spaces and the thresh is mainly white

    whitePixels = 0

    # count white pixels --> spaces with no photos in canvas

    for i in range(background.shape[0]):
        for j in range(background.shape[1]):
            for item in background[i][j]:
                if item == 255:
                    whitePixels -= 1

    return whitePixelsFactor * whitePixels + individualThreshFactor * threshAverage - penalizationCount*penalizationQty

def find_elite(population,fitness):

    idx = np.argmax(fitness)
    return population[idx], fitness[idx]

def mutate_polutation(population):

    n = len(population)
    d = len(population[0])

    vPopulation = np.zeros_like(population, float)

    for i in range(n):

        n1 = np.random.randint(n)
        n2 = np.random.randint(n)
        n3 = np.random.randint(n)
        while n1 == n2 or n1 == n3 or n2 == n3:
            n1 = np.random.randint(n)
            n2 = np.random.randint(n)
            n3 = np.random.randint(n)

        for k in range(d):
            vPopulation[i][k] = population[n1][k] + np.random.uniform(0, 2) * (population[n2][k] - population[n3][k])

            for j in range(2):
                if vPopulation[i][k][j] < 0:
                    vPopulation[i][k][j] = 0
                elif vPopulation[i][k][j] > backgroundArea:
                    vPopulation[i][k][j] = backgroundArea

            for j in range(2,4):
                if vPopulation[i][k][j] < 0.5:
                    vPopulation[i][k][j] = 0.5
                elif vPopulation[i][k][j] > 1:
                    vPopulation[i][k][j] = 1

            if vPopulation[i][k][4] < -math.pi/4:
                vPopulation[i][k][4] = -math.pi/4
            elif vPopulation[i][k][4] > math.pi/4:
                vPopulation[i][k][4] = math.pi/4

    return np.array(vPopulation)

def offspring(population, vPopulation, CR):

    n = len(population)
    d = len(population[0])
    items = len(population[0][0])

    uPopulation = np.zeros_like(population, float)

    for i in range(n):
        for k in range(d):
            l = np.random.randint(0, items - 1)
            for j in range(items):
                randFloat = np.random.random()
                if randFloat < CR or j == l:
                    uPopulation[i][k][j] = vPopulation[i][k][j]
                else:
                    uPopulation[i][k][j] = population[i][k][j]

    return uPopulation

def selection(population,fitness,uPopulation,uFitness):

    n = len(population)

    newPopulation = np.zeros_like(population,float)
    newFitness = []

    for i in range(n):

        if fitness[i] >= uFitness[i]:
            newPopulation[i] = population[i]
            newFitness.append(fitness[i])
        else:
            newPopulation[i] = uPopulation[i]
            newFitness.append(uFitness[i])

    return newPopulation, newFitness

# SHAPE = (HEIGHT, WIDTH)

img1 = cv2.imread('images/person1.bmp')
img2 = cv2.imread('images/person2.bmp')
img3 = cv2.imread('images/person3.bmp')
img4 = cv2.imread('images/tennis.bmp')
img5 = cv2.imread('images/sheep.bmp')

thresh1 = cv2.imread('thresh/person1.bmp')
thresh2 = cv2.imread('thresh/person2.bmp')
thresh3 = cv2.imread('thresh/person3.bmp')
thresh4 = cv2.imread('thresh/tennis.bmp')
thresh5 = cv2.imread('thresh/sheep.bmp')

images = []
images.append(img1)
images.append(img2)
images.append(img3)
images.append(img4)
images.append(img5)

thresh = []
thresh.append(thresh1)
thresh.append(thresh2)
thresh.append(thresh3)
thresh.append(thresh4)
thresh.append(thresh5)

backgroundArea = 0
scale_percent = 40

for i in range(len(images)):
    width = int(images[i].shape[1] * scale_percent / 100)
    height = int(images[i].shape[0] * scale_percent / 100)
    dim = (width, height)
    images[i] = cv2.resize(images[i], dim, interpolation=cv2.INTER_AREA)
    thresh[i] = cv2.resize(thresh[i], dim, interpolation=cv2.INTER_AREA)
    backgroundArea += images[i].shape[0]*images[i].shape[1]

backgroundArea /= 2
backgroundArea = int(math.sqrt(backgroundArea))

# ind = [y,x,escalaY,escalaX, rotacion]

n = 30
maxIter = 200
CR = 0.3
whitePixelsFactor = 0.5
individualThreshFactor = 0.5
threshAppearance = 0.7
minThreshAppearance = 0.6
penalizationQty = 10000

population = create_population(n)
fitness = get_fitness(population)
elite, fitnessElite = find_elite(population,fitness)

k = 0

while k < maxIter:

    print("k:", k, "f:", fitnessElite)

    vPopulation = mutate_polutation(population)
    uPopulation = offspring(population, vPopulation, CR)
    uFitness = get_fitness(uPopulation)
    population, fitness = selection(population, fitness, uPopulation, uFitness)
    elite, fitnessElite = find_elite(population, fitness)

    cv2.imwrite('results/image5_'+str(k)+'.png', create_image_from_individual(elite))
    threshMap, other, other2 = thresh_related_fitness(create_matrix_from_individual(elite),elite)

    k += 1

print("k:", k, "f:", fitnessElite)

cv2.imshow('final', create_image_from_individual(elite))
cv2.waitKey(0)
cv2.destroyAllWindows()
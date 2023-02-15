from ARC.apriori import Apriori
import time

start = time.time()
if __name__ == "__main__":
    arc = Apriori(filename='data.json', min_length=2)
    arc.apriori(excel=True)

end = time.time()
print(" \n", (end - start) * 10 ** 3, "ms")

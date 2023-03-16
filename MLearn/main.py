from arl.apriori import Apriori
from clustering.KMeans import KMeans
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from regression.Linear import Linear

start = time.time()
if __name__ == "__main__":
    # arc = Apriori(filename='data.json', min_length=2)
    # arc.apriori(excel=True)

    X_old = np.array([[37.51422, 55.679611],
                      [37.513505, 55.679965],
                      [37.528642, 55.688582],
                      [37.527154, 55.687055],
                      [37.525031, 55.686625],
                      [37.523491, 55.683753],
                      [37.522561, 55.685101],
                      [37.521183, 55.684327],
                      [37.518864, 55.682301],
                      [37.518408, 55.681713],
                      [37.519499, 55.68103],
                      [37.51528, 55.681214],
                      [37.515277, 55.680335],
                      [37.538153, 55.683988],
                      [37.536368, 55.682758],
                      [37.534194, 55.681229],
                      [37.528434, 55.677395],
                      [37.524959, 55.676671],
                      [37.522452, 55.677014],
                      [37.524548, 55.677579],
                      [37.52489, 55.67495],
                      [37.513381, 55.679956],
                      [37.519066, 55.67788],
                      [37.517325, 55.677375],
                      [37.516104, 55.679226],
                      [37.515823, 55.678876],
                      [37.514934, 55.679213],
                      [37.515122, 55.679782],
                      [37.521456, 55.675229],
                      [37.522172, 55.675836],
                      [37.522992, 55.676153],
                      [37.51921, 55.676629],
                      [37.530466, 55.681007],
                      [37.527169, 55.682987],
                      [37.525495, 55.682673],
                      [37.526667, 55.686127],
                      [37.525591, 55.683541],
                      [37.524806, 55.684651],
                      [37.529655, 55.681576],
                      [37.53143, 55.682817],
                      [37.530911, 55.683216],
                      [37.530268, 55.683503],
                      [37.528091, 55.681318],
                      [37.531607, 55.684696],
                      [37.528077, 55.682131],
                      [37.529173, 55.684075],
                      [37.528762, 55.68447],
                      [37.528135, 55.684769],
                      [37.52415, 55.6799],
                      [37.525108, 55.681839],
                      [37.523686, 55.68248],
                      [37.523256, 55.680862],
                      [37.521391, 55.683315],
                      [37.520448, 55.680741],
                      [37.521706, 55.680662],
                      [37.521858, 55.68009],
                      [37.52227, 55.681824],
                      [37.519094, 55.680327],
                      [37.521586, 55.682141],
                      [37.520908, 55.682524],
                      [37.52727, 55.678938],
                      [37.526358, 55.678478],
                      [37.525309, 55.678086],
                      [37.520395, 55.683716],
                      [37.519519, 55.683277],
                      [37.527492, 55.680432],
                      [37.525755, 55.679425],
                      [37.525926, 55.680743],
                      [37.522862, 55.679231],
                      [37.524155, 55.67919],
                      [37.524286, 55.678605],
                      [37.534554, 55.686902],
                      [37.533183, 55.686155],
                      [37.531756, 55.686903],
                      [37.531807, 55.686157],
                      [37.53091, 55.686225],
                      [37.530853, 55.685617],
                      [37.529959, 55.685701],
                      [37.52988, 55.685076],
                      [37.532139, 55.68812],
                      [37.530251, 55.688116],
                      [37.537462, 55.684996],
                      [37.533984, 55.684924],
                      [37.534045, 55.684289],
                      [37.533, 55.684342],
                      [37.533071, 55.683754],
                      [37.532059, 55.683814],
                      [37.536428, 55.685886],
                      [37.535404, 55.684957],
                      [37.53403, 55.685735],
                      [37.532629, 55.685222],
                      [37.533079, 55.691038],
                      [37.542851, 55.688829],
                      [37.541211, 55.686048],
                      [37.541212, 55.68906],
                      [37.538881, 55.689805],
                      [37.537218, 55.692034],
                      [37.536484, 55.687171],
                      [37.537198, 55.686633],
                      [37.538013, 55.686254],
                      [37.538888, 55.686668],
                      [37.539576, 55.687153],
                      [37.539428, 55.687557],
                      [37.538388, 55.687996],
                      [37.53565, 55.687613],
                      [37.53504, 55.687876],
                      [37.53429, 55.688388],
                      [37.535067, 55.688906],
                      [37.535794, 55.689232],
                      [37.536711, 55.689018],
                      [37.537562, 55.688441],
                      [37.5195, 55.68794],
                      [37.518339, 55.688612],
                      [37.509317, 55.684944],
                      [37.510136, 55.685452],
                      [37.532882, 55.694216],
                      [37.531668, 55.694873],
                      [37.528361, 55.694024],
                      [37.530806, 55.693009],
                      [37.516854, 55.703428],
                      [37.515325, 55.702417],
                      [37.514349, 55.704046],
                      [37.509616, 55.707899],
                      [37.50871, 55.707431],
                      [37.506151, 55.70676],
                      [37.505899, 55.706484],
                      [37.50665, 55.706016],
                      [37.507188, 55.705711],
                      [37.508851, 55.703938],
                      [37.509047, 55.70004],
                      [37.510718, 55.700461],
                      [37.511684, 55.699531],
                      [37.509846, 55.699056],
                      [37.508016, 55.703468],
                      [37.511294, 55.698209],
                      [37.512584, 55.697899],
                      [37.512649, 55.697546],
                      [37.51268, 55.697181],
                      [37.511762, 55.696722],
                      [37.505528, 55.702105],
                      [37.510213, 55.695852],
                      [37.504907, 55.697789],
                      [37.50691, 55.697664],
                      [37.507424, 55.69662],
                      [37.504517, 55.701699],
                      [37.502973, 55.699545],
                      [37.504039, 55.699586],
                      [37.505653, 55.699898],
                      [37.503379, 55.698308],
                      [37.505019, 55.698889],
                      [37.50277, 55.700712],
                      [37.502816, 55.696516],
                      [37.504609, 55.696271],
                      [37.503254, 55.69555],
                      [37.505021, 55.695236],
                      [37.501909, 55.700078],
                      [37.501984, 55.697683],
                      [37.501278, 55.696744],
                      [37.501211, 55.695996],
                      [37.499348, 55.696452],
                      [37.503889, 55.69325],
                      [37.50425, 55.693933],
                      [37.505606, 55.693471],
                      [37.505003, 55.692882],
                      [37.512632, 55.704435],
                      [37.496662, 55.695312],
                      [37.498316, 55.695109],
                      [37.498717, 55.694166],
                      [37.49671, 55.694314],
                      [37.500992, 55.693057],
                      [37.501704, 55.692208],
                      [37.503318, 55.692279],
                      [37.511309, 55.703731],
                      [37.510272, 55.702837],
                      [37.513481, 55.700443],
                      [37.510118, 55.704737],
                      [37.509512, 55.705136],
                      [37.507348, 55.701303],
                      [37.506364, 55.700755],
                      [37.509003, 55.701514],
                      [37.506532, 55.700168],
                      [37.508306, 55.700688],
                      [37.480575, 55.685094],
                      [37.480751, 55.68586],
                      [37.482488, 55.686269],
                      [37.49731, 55.70471],
                      [37.495002, 55.703153],
                      [37.498378, 55.691751],
                      [37.496988, 55.690968],
                      [37.495635, 55.690233],
                      [37.491073, 55.691562],
                      [37.490356, 55.690767],
                      [37.488981, 55.69059],
                      [37.489784, 55.690099],
                      [37.488402, 55.689868],
                      [37.492063, 55.690051],
                      [37.49282, 55.68881],
                      [37.495064, 55.688667],
                      [37.494997, 55.687975],
                      [37.492796, 55.68794],
                      [37.490619, 55.689495],
                      [37.493503, 55.693344],
                      [37.492753, 55.693221],
                      [37.496761, 55.692613],
                      [37.495361, 55.691856],
                      [37.494077, 55.691105],
                      [37.492376, 55.6924],
                      [37.491207, 55.692304],
                      [37.498201, 55.692535],
                      [37.496787, 55.691826],
                      [37.495453, 55.691062],
                      [37.494134, 55.690303],
                      [37.504976, 55.706019],
                      [37.498909, 55.703143],
                      [37.499265, 55.702639],
                      [37.498814, 55.701607],
                      [37.496882, 55.702464],
                      [37.507524, 55.704855],
                      [37.506625, 55.704341],
                      [37.502833, 55.703915],
                      [37.502812, 55.704902],
                      [37.50432, 55.705221],
                      [37.501338, 55.703792],
                      [37.485295, 55.686788],
                      [37.484521, 55.687451],
                      [37.483228, 55.687063],
                      [37.484509, 55.68575],
                      [37.501772, 55.675894],
                      [37.501222, 55.67625],
                      [37.501079, 55.675384],
                      [37.500696, 55.674619],
                      [37.500486, 55.675715],
                      [37.497274, 55.676276],
                      [37.497687, 55.676385],
                      [37.510934, 55.67827],
                      [37.510458, 55.677895],
                      [37.499326, 55.673843],
                      [37.499556, 55.675243],
                      [37.499014, 55.675491],
                      [37.497521, 55.674594],
                      [37.496528, 55.67465],
                      [37.505556, 55.673369],
                      [37.50611, 55.673067],
                      [37.506649, 55.67275],
                      [37.498072, 55.673157],
                      [37.494826, 55.673594],
                      [37.495136, 55.672517],
                      [37.49417, 55.672702],
                      [37.500942, 55.672958],
                      [37.499159, 55.671931],
                      [37.495621, 55.674215],
                      [37.501606, 55.672408],
                      [37.502753, 55.671796],
                      [37.504073, 55.671586],
                      [37.505739, 55.671671],
                      [37.50131, 55.671755],
                      [37.4968, 55.672415],
                      [37.502902, 55.671075],
                      [37.504466, 55.670848],
                      [37.506081, 55.670896],
                      [37.495505, 55.671725],
                      [37.507085, 55.672331],
                      [37.506951, 55.671401],
                      [37.507708, 55.67114],
                      [37.498012, 55.671206],
                      [37.498759, 55.670988],
                      [37.499534, 55.670724],
                      [37.500339, 55.670482],
                      [37.514592, 55.66824],
                      [37.513368, 55.66954],
                      [37.513748, 55.668811],
                      [37.512978, 55.667217],
                      [37.512118, 55.667647],
                      [37.51138, 55.666128],
                      [37.510219, 55.667493],
                      [37.510502, 55.666633],
                      [37.508795, 55.667972],
                      [37.508449, 55.668441],
                      [37.508548, 55.666045],
                      [37.517799, 55.670384],
                      [37.516808, 55.670836],
                      [37.516175, 55.669324],
                      [37.515007, 55.670517],
                      [37.515287, 55.669815],
                      [37.493888, 55.680297],
                      [37.486329, 55.677556],
                      [37.492486, 55.678272],
                      [37.485219, 55.677048],
                      [37.493531, 55.677558],
                      [37.491392, 55.6784],
                      [37.492098, 55.677618],
                      [37.492658, 55.679464],
                      [37.491136, 55.68033],
                      [37.491032, 55.677479],
                      [37.489975, 55.677851],
                      [37.492082, 55.676693],
                      [37.490629, 55.676814],
                      [37.48957, 55.67681],
                      [37.493919, 55.679226],
                      [37.490627, 55.675899],
                      [37.488329, 55.676961],
                      [37.48919, 55.676049],
                      [37.487981, 55.676243],
                      [37.487728, 55.675583],
                      [37.494999, 55.678568],
                      [37.488702, 55.678938],
                      [37.487911, 55.680003],
                      [37.493549, 55.67847],
                      [37.485977, 55.679011],
                      [37.513925, 55.677727],
                      [37.514553, 55.677258],
                      [37.507838, 55.667797],
                      [37.482106, 55.684975],
                      [37.482784, 55.685442],
                      [37.505703, 55.666883],
                      [37.506729, 55.668036],
                      [37.504922, 55.667402],
                      [37.509746, 55.665233],
                      [37.50409, 55.667925],
                      [37.507694, 55.669494],
                      [37.506916, 55.669306],
                      [37.505336, 55.669737],
                      [37.504521, 55.669735],
                      [37.503755, 55.669737],
                      [37.507228, 55.665676],
                      [37.502769, 55.66853],
                      [37.499754, 55.669818],
                      [37.499047, 55.67005],
                      [37.498293, 55.67028],
                      [37.497198, 55.670823],
                      [37.492413, 55.672155],
                      [37.506843, 55.666249],
                      [37.494413, 55.675484],
                      [37.49352, 55.675454],
                      [37.492616, 55.675438],
                      [37.491438, 55.672839],
                      [37.491611, 55.673694],
                      [37.490533, 55.67318],
                      [37.491511, 55.674987],
                      [37.491207, 55.674579],
                      [37.489614, 55.673832],
                      [37.507694, 55.667028],
                      [37.48871, 55.673888],
                      [37.487713, 55.674673],
                      [37.489176, 55.675149],
                      [37.48344, 55.679766],
                      [37.483407, 55.680301],
                      [37.48298, 55.681068],
                      [37.484714, 55.682235],
                      [37.482178, 55.682068],
                      [37.482862, 55.682806],
                      [37.481226, 55.683371],
                      [37.481895, 55.684088],
                      [37.487757, 55.68913],
                      [37.518154, 55.671423],
                      [37.513424, 55.674636],
                      [37.51243, 55.675205],
                      [37.511954, 55.676025],
                      [37.515265, 55.672514],
                      [37.510502, 55.676336],
                      [37.513253, 55.673129],
                      [37.512958, 55.672806],
                      [37.51069, 55.67337],
                      [37.510365, 55.673071],
                      [37.498412, 55.680782],
                      [37.511101, 55.673718],
                      [37.497131, 55.681001],
                      [37.495887, 55.681255],
                      [37.494656, 55.681017],
                      [37.509973, 55.674203],
                      [37.495999, 55.682072],
                      [37.509513, 55.670642],
                      [37.509335, 55.670368],
                      [37.509011, 55.671835],
                      [37.508993, 55.669425],
                      [37.508652, 55.672239],
                      [37.512481, 55.670837],
                      [37.512303, 55.670562],
                      [37.5114, 55.671185],
                      [37.511222, 55.67091],
                      [37.510321, 55.671523],
                      [37.510147, 55.671128],
                      [37.510467, 55.670257],
                      [37.510293, 55.669863],
                      [37.494579, 55.681879],
                      [37.493542, 55.681569],
                      [37.508797, 55.675063],
                      [37.507903, 55.675458],
                      [37.508331, 55.674689],
                      [37.507245, 55.67518],
                      [37.516901, 55.673279],
                      [37.50763, 55.674414],
                      [37.506737, 55.674804],
                      [37.491308, 55.684573],
                      [37.507048, 55.674085],
                      [37.492403, 55.685241],
                      [37.506125, 55.674493],
                      [37.493804, 55.685849],
                      [37.504197, 55.677626],
                      [37.516094, 55.671521],
                      [37.516193, 55.671904],
                      [37.516472, 55.672136],
                      [37.490533, 55.685201],
                      [37.503491, 55.678148],
                      [37.489401, 55.685611],
                      [37.5036, 55.677299],
                      [37.502902, 55.677823],
                      [37.503015, 55.676971],
                      [37.502315, 55.677496],
                      [37.514109, 55.674],
                      [37.502421, 55.676629],
                      [37.501719, 55.677156],
                      [37.500423, 55.678994],
                      [37.501571, 55.678806],
                      [37.501816, 55.67843],
                      [37.499541, 55.679143],
                      [37.513991, 55.671575],
                      [37.513658, 55.671333],
                      [37.498555, 55.677799],
                      [37.499216, 55.677587],
                      [37.499383, 55.677058],
                      [37.497293, 55.677903],
                      [37.497555, 55.67929],
                      [37.496631, 55.679647],
                      [37.494632, 55.679799],
                      [37.489143, 55.68256],
                      [37.486511, 55.681137],
                      [37.491148, 55.681665],
                      [37.486607, 55.68628],
                      [37.487749, 55.684815],
                      [37.486704, 55.684174],
                      [37.516296, 55.681658],
                      [37.544005, 55.688135],
                      [37.50605, 55.699453],
                      [37.505894, 55.675544],
                      [37.50612, 55.675904],
                      [37.528882, 55.688741],
                      [37.501378, 55.697709],
                      [37.513925, 55.677722],
                      [37.5061, 55.67092],
                      [37.489724, 55.6901],
                      [37.499717, 55.697352],
                      [37.536217, 55.685971],
                      [37.49615, 55.693453],
                      [37.522859, 55.681373],
                      [37.504591, 55.674519],
                      [37.484083, 55.686174],
                      [37.535826, 55.686168],
                      [37.500972, 55.699766],
                      [37.537547, 55.691853],
                      [37.501237, 55.674936],
                      [37.524822, 55.674927],
                      [37.513924, 55.677723],
                      [37.543705, 55.688293],
                      [37.501656, 55.672432],
                      [37.4863, 55.67545],
                      [37.493782, 55.692173],
                      [37.503784, 55.702442],
                      [37.500206, 55.697464],
                      [37.49096976, 55.68622382],
                      [37.52087411, 55.67779807],
                      [37.52933729, 55.68044484],
                      [37.52165499, 55.67745173],
                      [37.49366973, 55.68453628],
                      [37.49569985, 55.69484247],
                      [37.51224687, 55.67634642],
                      [37.53065932, 55.68983388],
                      [37.53463657, 55.69083205],
                      [37.52420144, 55.68627765],
                      [37.495546, 55.67169573],
                      [37.53291358, 55.68033821],
                      [37.54212608, 55.68929789],
                      [37.50453196, 55.70148308],
                      [37.4935838, 55.69329931],
                      [37.49094673, 55.69044611],
                      [37.53675803, 55.68575337],
                      [37.50735498, 55.67611692],
                      [37.52058131, 55.68363114],
                      [37.48716251, 55.68678214],
                      [37.52432712, 55.6807129],
                      [37.51657276, 55.68166252],
                      [37.50574738, 55.67621642],
                      [37.495455, 55.675753],
                      [37.50790225, 55.67272559],
                      [37.49503754, 55.69350281],
                      [37.5075311, 55.677704],
                      [37.53506754, 55.69149555],
                      [37.49131197, 55.69042459],
                      [37.50552388, 55.67717489],
                      [37.49347743, 55.69265413],
                      [37.5354567, 55.69249258],
                      [37.53444448, 55.69360079],
                      [37.49637895, 55.69445851],
                      [37.5339882, 55.69332385],
                      [37.51327005, 55.66667516],
                      [37.53502122, 55.6933963],
                      [37.50554247, 55.67658235],
                      [37.53260999, 55.68030699]])
    # X1 = np.random.normal(loc=[0, -10], size=(100, 2))
    # X2 = np.random.normal(loc=[-10, 0], size=(100, 2))
    # X3 = np.random.normal(loc=[0, 0], size=(100, 2))
    # X = np.vstack((X1, X2, X3))

    # km = KMeans(10)
    # markers = km.fit_predict(X_old)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # axs[1].scatter(X[:, 0], X[:, 1], c=markers)
    # axs[0].scatter(X[:, 0], X[:, 1], marker='o')
    # plt.show()
    df_train = pd.read_excel('test.xlsx', engine='openpyxl')

    # X = df_train['square'].to_numpy()
    # Y = df_train['clusters'].to_numpy()
    # X = np.array([1, 1.2, 1.6, 1.78, 2, 2.3, 2.4, 3, 3.3, 4, 4.1, 4.12, 4.34, 5, 5.3, 5.6, 6])
    # Y = np.array([0.8, 1, 0.9, 1.0, 1.2, 1.1, 1.6, 1.7, 2.0, 2.1, 2.15, 2.22, 2.45, 2.6, 2.12, 2.45, 2.3])
    X = np.array([1, 3])
    Y = np.array([2, 3])

    regression = Linear(learning_rate=0.1, max_iter=100, optimizer_name='GD') # learning_rate=0.05, max_iter=100, optimizer_name='Adam'
    # regression.fit(X=df_train['square'].to_numpy(), Y=df_train['clusters'].to_numpy())

    regression.fit(X=X, Y=Y)
    x_loss = np.arange(len(regression.loss_history))

    fig, axe = plt.subplots(1, 2, figsize=(15, 6))
    axe[0].scatter(X, Y, marker='o', alpha=0.8)
    axe[0].plot(X, regression.pred, 'r')
    axe[1].plot(x_loss, regression.loss_history)
    plt.show()

    print('y = ' + str(regression.weight_) + ' * x + ' + str(regression.bias_))
    print(regression.predict(7))

end = time.time()
print(" \n", (end - start) * 10 ** 3, "ms")

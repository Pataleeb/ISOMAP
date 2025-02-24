from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components, shortest_path
import numpy as np
from scipy.io import loadmat
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.linalg import eigh
import math

class OrderOfFaces:
    def __init__(self, images_path = 'data/isomap.mat'):
        mat_data=loadmat(images_path)
        self.data=mat_data['images'].T
        assert self.data.shape == (698,4096), f"Unexpected shape of {self.data.shape}"
        #raise NotImplementedError("Not Implemented")

    def get_adjacency_matrix(self, epsilon):
        distance_matrix = cdist(self.data, self.data)
        #epsilon=np.percentile(distance_matrix, 10)
        adjacency_matrix = np.where(distance_matrix <= epsilon, 1,0)
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
        return adjacency_matrix

    def get_best_epsilon(self,minep=5.0,maxep=50.0,step=1.0):


        for epsilon in np.arange(minep,maxep,step):
            adjacency_matrix = self.get_adjacency_matrix(epsilon)
            n_components,_=connected_components(adjacency_matrix,directed=False)

            if n_components==1:
                return epsilon
        return None

    def isomap(self, epsilon):
        adjacency_matrix = self.get_adjacency_matrix(epsilon)
        shortest_distance=shortest_path(adjacency_matrix,directed=False,method='D')

        m=shortest_distance.shape[0]
        H=np.eye(m)-np.ones((m,m))/m
        D_sq=shortest_distance**2
        C=-0.5*H @ D_sq @ H

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx=np.argsort(eigenvalues)[::-1] [:2]
        eigenvectors=eigenvectors[:,idx]
        eigenvalues=eigenvalues[idx]

        embedding=eigenvectors*np.sqrt(eigenvalues)
        return embedding

    def graph(self, epsilon):
        adjacency_mat=self.get_adjacency_matrix(epsilon)

        graph=nx.from_numpy_array(adjacency_mat)
        pos=nx.spring_layout(graph,seed=6740)

        plt.figure(figsize=(10,10))
        nx.draw(graph,pos=pos,with_labels=False,node_size=10,alpha=0.7,edge_color='gray')
        nodes=np.array(list(pos.keys()))
        sample_nodes=np.linspace(0,len(nodes)-1,5, dtype=int)
        sample_nodes=nodes[sample_nodes]

        nx.draw_networkx_nodes(graph, pos, nodelist=sample_nodes, node_size=50, node_color='red')


        for node in sample_nodes:
            x,y=pos[node]
            plt.text(x,y,str(node),fontsize=10,color='black',fontweight='bold',ha='center',va='center')
        plt.title(f'Nearest Neighbor Graph (epsilon={epsilon})')
        plt.axis('equal')
        plt.show()

        for node in sample_nodes:
            plt.figure()
            plt.imshow(self.data[node].reshape(64,64),cmap='gray')
            plt.title(f'Node {node}')
            plt.axis('off')
            plt.show()


faces=OrderOfFaces(images_path='data/isomap.mat')
best_epsilon=faces.get_best_epsilon()
print("Best Epsilon:",best_epsilon)

faces.graph(best_epsilon)

#### Q2
def plot_scatter_with_images(embedding, data, specific_points, image_shape=(64, 64), zoom=0.7):

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5,label='All Points')

    for i in specific_points:
        image = data[i].reshape(image_shape)
        imagebox = OffsetImage(image, cmap='gray', zoom=zoom)
        ab = AnnotationBbox(imagebox, (embedding[i, 0], embedding[i, 1]), frameon=False)
        ax.add_artist(ab)


    ax.set_title("ISOMAP Embedding with Images")
    plt.xlabel("D 1")
    plt.ylabel("D 2")
    plt.legend()
    plt.grid(False)
    plt.show()



embedding = faces.isomap(best_epsilon)

specific_points = [10, 100, 200, 300,400,600]

plot_scatter_with_images(embedding, faces.data, specific_points, image_shape=(64, 64))

###Tune the bandwidth
distance_matrix = cdist(faces.data, faces.data)
epsilon = np.percentile(distance_matrix, 11)


####Q3
class OrderOfFaces:
    def __init__(self, images_path='data/isomap.mat'):
        mat_data = loadmat(images_path)
        self.data = mat_data['images'].T
        print(f"Shape of self.data: {self.data.shape}")

    def pca(self,num_components=2):
        mat_data=self.data
        m,n=mat_data.shape

        stdmat=np.std(mat_data,axis=0)
        stdmat=np.where(stdmat==0,1e-10,stdmat)
        mat_data=mat_data/stdmat

        mu=np.mean(mat_data,axis=0)
        xc=mat_data - mu
        xc=np.nan_to_num(xc)

        

        C=np.dot(xc.T,xc)/m
        S,W=eigh(C)
        S,W=S[-num_components:],W[:,-num_components:]

        dim1 = xc @ W[:, 0] / math.sqrt(S[0])
        dim2 = xc @ W[:, 1] / math.sqrt(S[1])

        embedding = np.vstack([dim1, dim2]).T
        return embedding


    def plot_pca_with_images(self, embedding, specific_points, image_shape=(64, 64), zoom=0.7):
        plt.figure(figsize=(10, 10))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, label="Data Points")

        for i in specific_points:
            image = self.data[i,:].reshape(image_shape)
            imagebox = OffsetImage(image, cmap='gray', zoom=zoom)
            ab = AnnotationBbox(imagebox, (embedding[i, 0], embedding[i, 1]), frameon=False)
            plt.gca().add_artist(ab)

        plt.title('PCA Embedding with Images')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid()
        plt.legend()
        plt.show()


faces = OrderOfFaces(images_path='data/isomap.mat')

pca_embedding = faces.pca()

specific_points = [10, 100, 200, 300, 400, 600]

faces.plot_pca_with_images(pca_embedding, specific_points, image_shape=(64, 64))

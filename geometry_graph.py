import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from igraph import Graph
import igraph

def geometry_graph(img, disc=1, degree_range=None, area_range=None, return_largest_component=False,
                   return_label_img=False):
    
    '''Create adjacency graph between white components in `img`.
    
    Parameters
    ----------
    img : numpy array
        Binary imagem 
    disc : discretization used to adjust for image
    degree_range : tuple of ints
        Only nodes having degree within this range will be returned
    area_range : tuple of floats
        Only regions having area within this range will be returned. NOT IMPLEMENTED
    return_largest_component : bool
        Whether to return only the largest connected component
    return_label_img : bool
        If True, also returns an image containing the labels of the components
        
    Returns
    -------
    g_return : igraph graph
        The geometry adjacency graph. Nodes have attribute `pos` indicating the center of
        mass of the respective region and also attribute `idx` indicating the index of the
        node in img `img_label`
    img_label : numpy array
        Returned if `return_label_img` is True. It is an image containing the labels of the
        conencted components
    '''
    
    if degree_range is None:
        degree_range = (0, np.inf)
    if area_range is None:
        area_range = (0, np.inf)
    
    img_label, num_comp = ndi.label(img)
    img_dist, (rows, cols) = ndi.distance_transform_edt(1-img, return_indices=True)
    # Expand indices of components in `img_label`. 
    img_label_expanded = img_label[rows, cols]
    
    # Create edge list of the graph
    edges = set(zip(img_label_expanded[1:].ravel(), img_label_expanded[0:-1].ravel()))
    edges = edges or set(zip(img_label_expanded[:,1:].ravel(), img_label_expanded[:,0:-1].ravel()))

    # Remove unnecessary edges
    unique_edges = set()
    for edge in edges:
        if edge[0]!=edge[1]:
            if edge[0]<edge[1]:
                unique_edges.add(edge)
            elif (edge[1], edge[0]) not in unique_edges:
                unique_edges.add((edge[1], edge[0]))
                
    cm = ndi.center_of_mass(img, img_label, range(0, num_comp+1))
    cm = np.round(cm).astype(int)
    # Change coordinate system from cartesian to rows and columns
    # areas=[np.round(np.count_nonzero(img_label==l)*(1/disc)**2).astype(int) for l in range(0,np.max(img_label)+1)]
    boxes = ndi.find_objects(label+1)
    factor=[]
    areas=[]

    for l in range(0,np.max(label)+1):
        temp_img=img[boxes[l]]
        x,y=temp_img.shape
        area=np.count_nonzero(temp_img)
        radius=np.sqrt(x**2+y**2)//2+1;
        factor.append(area/(radius**2*np.pi))
        areas.append(np.round(area*(1/disc)**2).astype(int))
        
    cm = cm[:,::-1]
    cm = list(zip(cm[:,0], cm[:,1]))                       # igraph usually prefers list of tuples instead of numpy array
                
    g = Graph(n=num_comp+1, edges=list(unique_edges))      # num_comp+1 because node 0 is not used
    g.vs['areas'] = areas
    g.vs['pos'] = cm
    g.vs['form_factor'] = factor
    g.vs['idx'] = range(num_comp+1)
    
    # Filter nodes
    nodes_to_keep = g.vs.select(_degree_ge=degree_range[0], _degree_le=degree_range[1])
    g_return = g.subgraph(nodes_to_keep)
    
    if area_range:
        a1,a2=area_range
        nodes_to_keep = g_return.vs.select(lambda x:x["areas"]>=a1 and x["areas"]<=a2)
        g_return = g_return.subgraph(nodes_to_keep)
    
    if return_largest_component:
        g_return = g_return.clusters(igraph.WEAK).giant()
    
    if return_label_img:
        return g_return, img_label
    else:
        return g_return
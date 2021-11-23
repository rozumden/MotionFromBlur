import kaolin
from models.initial_mesh import generate_initial_mesh

def load_obj(path):
    return kaolin.io.obj.import_mesh(path,with_materials=True)

def prepare_vertices(vertices, faces, camera_rot, camera_trans, camera_proj):
    vertices_camera = kaolin.render.camera.rotate_translate_points(vertices, camera_rot, camera_trans)
    vertices_image = kaolin.render.camera.perspective_camera(vertices_camera, camera_proj)
    face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, faces)
    face_normals = kaolin.ops.mesh.face_normals(face_vertices_camera, unit=True)
    return face_vertices_camera, face_vertices_image, face_normals

def write_obj_traj(points, name):
	file = open(name,"w")
	file.write("o FMO\n")
	for ver in points:
		file.write("v {:.6f} {:.6f} {:.6f} \n".format(ver[0],ver[1],ver[2]))
	file.write("l ")
	for pi in range(len(points)):
		file.write("{} ".format(pi+1))
	file.write("\n")
	file.close() 

def write_obj_traj_exp(points, name):
	file = open(name,"w")
	file.write("o FMO\n")
	for ver in points:
		file.write("v {:.6f} {:.6f} {:.6f} \n".format(ver[0],ver[1],ver[2]))
	pi = 1
	while pi <= len(points):
		file.write("l {} {}\n".format(pi,pi+1))
		pi += 2
	file.close() 

def write_obj_mesh(vertices, faces, face_features, name):
	file = open(name,"w")
	file.write("mtllib model.mtl\n")
	file.write("o FMO\n")
	for ver in vertices:
		file.write("v {:.6f} {:.6f} {:.6f} \n".format(ver[0],ver[1],ver[2]))
	for ffeat in face_features:
		for feat in ffeat:
			if len(feat) == 3:
				file.write("vt {:.6f} {:.6f} {:.6f} \n".format(feat[0],feat[1],feat[2]))
			else:
				file.write("vt {:.6f} {:.6f} \n".format(feat[0],feat[1]))
	file.write("usemtl Material.002\n")
	file.write("s 1\n")
	for fi in range(faces.shape[0]):
		fc = faces[fi]+1
		ti = 3*fi + 1
		file.write("f {}/{} {}/{} {}/{}\n".format(fc[0],ti,fc[1],ti+1,fc[2],ti+2))
	file.close() 


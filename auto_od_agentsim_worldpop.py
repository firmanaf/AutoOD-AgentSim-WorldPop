# -*- coding: utf-8 -*-
"""
QGIS Processing Toolbox
Name: Automatic OD from Boundary + Road Network + WorldPop Raster
Short description:
Builds an Origin–Destination (OD) matrix and sampled individual trips automatically using WorldPop population within a boundary. Travel impedance is computed via road-network distance when feasible, with a fallback to Euclidean/geodesic distance. Outputs include zone polygons with population, the OD matrix CSV, weighted OD lines, and a sampled individuals CSV.

Usage notes:
1. Ensure the road layer is in a projected CRS (meters) so network distances are valid. Otherwise, the algorithm warns and uses geodesic distance instead.
2. WorldPop is expected to have a single band representing population. Values are treated as persons per pixel and summed per zone.
3. The zone grid size controls zone resolution and is clipped by the boundary.
4. A gravity model is used to generate the OD matrix. Parameters are configurable.
5. Person sampling draws origin and destination points inside zone polygons.

Authored by: Firman Afrianto; refactored by assistant
"""

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDistance,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterVectorDestination,
    QgsFeature,
    QgsFields,
    QgsField,
    edit,
    QgsGeometry,
    QgsWkbTypes,
    QgsVectorLayer,
    QgsCoordinateTransform,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsSpatialIndex,
    QgsPointXY,
    QgsVectorFileWriter
)

import math
import os
import json
import csv
import numpy as np

# Rasterio is used for raster masking and value reading
try:
    import rasterio
    from rasterio.mask import mask as rio_mask
except Exception:
    rasterio = None

# NetworkX is used for optional network distance
try:
    import networkx as nx
except Exception:
    nx = None


def haversine_m(p1, p2):
    # p1 and p2 in degrees
    R = 6371000.0
    lon1, lat1 = math.radians(p1.x()), math.radians(p1.y())
    lon2, lat2 = math.radians(p2.x()), math.radians(p2.y())
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class ODAutomatWorldPop(QgsProcessingAlgorithm):
    INPUT_BOUNDARY = 'INPUT_BOUNDARY'
    INPUT_ROADS = 'INPUT_ROADS'
    INPUT_WORLDPOP = 'INPUT_WORLDPOP'
    GRID_SIZE = 'GRID_SIZE'
    MIN_POP = 'MIN_POP'
    USE_NETWORK = 'USE_NETWORK'
    AGENT_NUM = 'AGENT_NUM'
    LAMBDA = 'LAMBDA'
    ALPHA = 'ALPHA'
    BETA = 'BETA'
    GAMMA = 'GAMMA'
    MIN_TRIPS = 'MIN_TRIPS'
    TOPN_PER_ORIGIN = 'TOPN_PER_ORIGIN'
    OUTPUT_ZONES = 'OUTPUT_ZONES'
    OUTPUT_OD = 'OUTPUT_OD'
    OUTPUT_EDGES = 'OUTPUT_EDGES'
    OUTPUT_PERSONS = 'OUTPUT_PERSONS'
    RANDOM_SEED = 'RANDOM_SEED'
    HIDE_INTRA = 'HIDE_INTRA'

    # Opsi B: render GIF langsung
    ANIM_ENABLE = 'ANIM_ENABLE'
    ANIM_DURATION = 'ANIM_DURATION_S'
    ANIM_FPS = 'ANIM_FPS'
    ANIM_SUBSAMPLE = 'ANIM_SUBSAMPLE'
    ANIM_GIF_PATH = 'ANIM_GIF_PATH'
    ANIM_ON_ROAD = 'ANIM_ON_ROAD'

    # --- Added for post-processing ---
    POST_ENABLE = 'POST_ENABLE'
    MAIN_MIN_TRIPS_EDGE = 'MAIN_MIN_TRIPS_EDGE'
    MAIN_TOPK_EDGES = 'MAIN_TOPK_EDGES'
    MAIN_SMOOTH_TOL = 'MAIN_SMOOTH_TOL'
    MAIN_DISSOLVE_BUFFER = 'MAIN_DISSOLVE_BUFFER'
    CLUS_EPS_M = 'CLUS_EPS_M'
    CLUS_MINPTS = 'CLUS_MINPTS'
    OUTPUT_MAIN_ROUTES = 'OUTPUT_MAIN_ROUTES'
    OUTPUT_ORIGIN_CLUSTERS = 'OUTPUT_ORIGIN_CLUSTERS'
    OUTPUT_DEST_CLUSTERS = 'OUTPUT_DEST_CLUSTERS'
    OUTPUT_ODMID_CLUSTERS = 'OUTPUT_ODMID_CLUSTERS'
    OUTPUT_ORIGIN_HULLS = 'OUTPUT_ORIGIN_HULLS'
    OUTPUT_DEST_HULLS = 'OUTPUT_DEST_HULLS'
    OUTPUT_ODMID_HULLS = 'OUTPUT_ODMID_HULLS'

    def tr(self, string):
        return QCoreApplication.translate('ODAutomatWorldPop', string)

    def createInstance(self):
        return ODAutomatWorldPop()

    def name(self):
        return 'auto_od_agentsim_worldpop'

    def displayName(self):
        return self.tr('AutoOD & AgentSim from WorldPop')

    def shortHelpString(self):
        return self.tr(
            "<p><i>Created by</i> <b>FIRMAN AFRIANTO</b></p>"
            "<p>Builds a <b>zone grid</b> inside the boundary, aggregates WorldPop population per zone, "
            "creates a <b>gravity model based OD matrix</b>, and <b>samples individual trips</b>. "
            "Travel impedance uses the <b>road network</b> when available, otherwise falls back to geodesic distance.</p>"

            "<p><b>What it does</b></p>"
            "<ul>"
            "<li>Generates zone grid from the boundary, clips to it, and sums WorldPop per cell.</li>"
            "<li>Computes OD matrix with exponential or power decay per parameters.</li>"
            "<li>Filters OD flows by minimum trips and Top N per origin.</li>"
            "<li>Creates weighted OD lines and a sampled individuals CSV.</li>"
            "<li><i>Optional Post processing</i>: computes zone trip <i>generation</i> and <i>attraction</i>, "
            "extracts main routes from weighted edges, and runs DBSCAN on origins, destinations, and edge midpoints.</li>"
            "<li><i>Optional GIF rendering</i>: animates sampled trips with <b>boundary (red)</b> and <b>road network (grey)</b> "
            "drawn as faded background layers.</li>"
            "</ul>"

            "<p><b>Inputs</b></p>"
            "<ul>"
            "<li><b>Boundary polygons</b> as study area.</li>"
            "<li><b>Road network</b> line layer in a projected CRS (meters) for network distance and as background in GIF.</li>"
            "<li><b>WorldPop raster</b> single band with population per pixel.</li>"
            "</ul>"

            "<p><b>Key Parameters</b></p>"
            "<ul>"
            "<li><b>Zone grid size</b>: cell size of the zone grid.</li>"
            "<li><b>Zone population threshold</b>: minimum population for a cell to become a zone.</li>"
            "<li><b>Use network distance</b>: prefer network distance when possible, "
            "otherwise Euclidean or geodesic distance.</li>"
            "<li><b>Number of individuals</b>: number of agents for trip sampling.</li>"
            "<li><b>Gravity model</b>: "
            "Alpha (origin exponent), Beta (destination exponent), "
            "Lambda (exponential decay rate), "
            "Gamma (distance power when using power decay; set 0 to use exponential).</li>"
            "<li><b>Filters</b>: Minimum trips to drop tiny flows, "
            "Top N destinations per origin to cap targets.</li>"
            "</ul>"

            "<p><b>Post processing Options</b></p>"
            "<ul>"
            "<li><b>Enable Post Processing</b> toggles all extra steps.</li>"
            "<li><b>Main Routes</b>: minimum trips threshold, Top K edges, smoothing tolerance, "
            "and dissolve buffer.</li>"
            "<li><b>DBSCAN</b>: EPS in meters and minPts for clustering origins, destinations, and midpoints.</li>"
            "</ul>"

            "<p><b>GIF Options</b></p>"
            "<ul>"
            "<li><b>Enable GIF creation</b>: toggles animation rendering.</li>"
            "<li><b>Duration</b>, <b>FPS</b>, and <b>Subsample agents</b> control playback quality.</li>"
            "<li>GIF shows sampled trips over time with <b>boundary (red outline, faded)</b> and "
            "<b>road network (grey lines, faded)</b> as background context.</li>"
            "</ul>"

            "<p><b>Outputs</b></p>"
            "<ul>"
            "<li><b>Zones with population</b> polygon layer with attributes pop, gen, att, net_ga, ratio_ga.</li>"
            "<li><b>OD matrix CSV</b> with columns i, j, trips.</li>"
            "<li><b>Weighted OD lines</b> line layer from i to j with trips weight.</li>"
            "<li><b>Sampled individuals CSV</b> with origin and destination coordinates per agent.</li>"
            "<li><i>If post processing is enabled</i>: "
            "<b>Main routes</b> line layer, "
            "<b>origin clusters</b>, <b>destination clusters</b>, <b>midpoint clusters</b>, "
            "and cluster <b>hulls</b> for each.</li>"
            "<li><i>If GIF rendering is enabled</i>: animated GIF file with moving agents over boundary and road background.</li>"
            "</ul>"

            "<p><b>Notes</b></p>"
            "<ul>"
            "<li><b>CRS</b>: use a projected CRS in meters for valid network distances. "
            "If the road graph is empty or invalid the tool falls back to geodesic distance.</li>"
            "<li><b>Large zone counts</b>: very large numbers of zones may trigger a fallback to geodesic distance "
            "to preserve memory stability.</li>"
            "<li><b>WorldPop</b>: values are treated as population per pixel and summed with all touched mode.</li>"
            "<li><b>Dependencies</b>: rasterio is required for raster reading and masking. "
            "networkx is optional for network distance. "
            "imageio and matplotlib are required for GIF rendering.</li>"
            "</ul>"
            
            "<p><b>Dependencies</b></p>"
            "<ul>"
            "<li><b>numpy</b> (required): matrix ops & sampling</li>"
            "<li><b>rasterio</b> (required): read & mask WorldPop raster</li>"
            "<li><b>networkx</b> (optional): road-network distance; fallback to Euclidean/geodesic if missing</li>"
            "<li><b>matplotlib</b> (optional): rendering of animated GIF</li>"
            "<li><b>imageio</b> (optional): writing animated GIF</li>"
            "</ul>"
            
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_BOUNDARY, self.tr('Boundary polygons'), [QgsProcessing.TypeVectorPolygon]))

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_ROADS, self.tr('Road network (lines)'), [QgsProcessing.TypeVectorLine]))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_WORLDPOP, self.tr('WorldPop raster')))

        self.addParameter(QgsProcessingParameterDistance(
            self.GRID_SIZE, self.tr('Zone grid size'), defaultValue=1000, parentParameterName=self.INPUT_BOUNDARY))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_POP, self.tr('Zone population threshold'), QgsProcessingParameterNumber.Double, 10.0))

        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_NETWORK, self.tr('Use network distance when available'), True))

        self.addParameter(QgsProcessingParameterNumber(
            self.AGENT_NUM, self.tr('Number of individuals to sample'), QgsProcessingParameterNumber.Integer, 10000))

        self.addParameter(QgsProcessingParameterNumber(
            self.LAMBDA, self.tr('Lambda (impedance)'), QgsProcessingParameterNumber.Double, 0.001))

        self.addParameter(QgsProcessingParameterNumber(
            self.ALPHA, self.tr('Alpha (origin exponent)'), QgsProcessingParameterNumber.Double, 1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.BETA, self.tr('Beta (destination exponent)'), QgsProcessingParameterNumber.Double, 1.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.GAMMA, self.tr('Gamma (distance power; set 0 to use exponential)'), QgsProcessingParameterNumber.Double, 0.0))

        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_ZONES, self.tr('Zones with population')))

        # Filtering controls
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_TRIPS, self.tr('Minimum trips to keep (filter small flows)'), QgsProcessingParameterNumber.Double, 0.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.TOPN_PER_ORIGIN, self.tr('Top-N destinations per origin (0 = all)'), QgsProcessingParameterNumber.Integer, 0))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_OD, self.tr('OD matrix CSV'), 'CSV files (*.csv)'))

        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_EDGES, self.tr('Weighted OD lines')))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT_PERSONS, self.tr('Sampled individuals CSV'), 'CSV files (*.csv)'))

        # --- Added for post-processing ---
        from qgis.core import QgsProcessingParameterBoolean as _Bool
        self.addParameter(_Bool(
            self.POST_ENABLE,
            self.tr('Enable Post-Processing (Generation/Attraction, Main Routes, DBSCAN)'),
            False
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAIN_MIN_TRIPS_EDGE,
            self.tr('[Main Routes] Minimum trips threshold for edges'),
            QgsProcessingParameterNumber.Double, 1.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAIN_TOPK_EDGES,
            self.tr('[Main Routes] Top-K edges for main routes'),
            QgsProcessingParameterNumber.Integer, 500
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAIN_SMOOTH_TOL,
            self.tr('[Main Routes] Smoothing tolerance (meters)'),
            QgsProcessingParameterNumber.Double, 50.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAIN_DISSOLVE_BUFFER,
            self.tr('[Main Routes] Dissolve buffer (meters)'),
            QgsProcessingParameterNumber.Double, 60.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.CLUS_EPS_M,
            self.tr('[DBSCAN] eps (meters)'),
            QgsProcessingParameterNumber.Double, 200.0
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.CLUS_MINPTS,
            self.tr('[DBSCAN] minPts'),
            QgsProcessingParameterNumber.Integer, 15
        ))

        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_MAIN_ROUTES,
            self.tr('[Main Routes] Main routes output')
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_ORIGIN_CLUSTERS,
            self.tr('[DBSCAN] Clustered origin points')
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_DEST_CLUSTERS,
            self.tr('[DBSCAN] Clustered destination points')
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_ODMID_CLUSTERS,
            self.tr('[DBSCAN] Clustered edge midpoints')
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_ORIGIN_HULLS,
            self.tr('[DBSCAN] Origin hulls (per cluster)')
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_DEST_HULLS,
            self.tr('[DBSCAN] Destination hulls (per cluster)')
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_ODMID_HULLS,
            self.tr('[DBSCAN] Midpoint hulls (per cluster)')
        ))
       
        # Reproducibility & intra-flow

        self.addParameter(QgsProcessingParameterNumber(
            self.RANDOM_SEED, self.tr('Random seed for reproducibility (-1 for random)'), 
            QgsProcessingParameterNumber.Integer, -1))

        self.addParameter(QgsProcessingParameterBoolean(
            self.HIDE_INTRA, self.tr('Hide intra-zone flows (i=j)'), False))

        # Option B: direct GIF animation output
        self.addParameter(QgsProcessingParameterBoolean(
            self.ANIM_ENABLE, self.tr('[Anim GIF] Enable GIF creation'), False))
        
        self.addParameter(_Bool(
            self.ANIM_ON_ROAD,
            self.tr('[Anim GIF] Constrain agent motion to road network'),
            False
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.ANIM_DURATION, self.tr('[Anim GIF] Animation duration (seconds)'), 
            QgsProcessingParameterNumber.Double, 20.0))

        self.addParameter(QgsProcessingParameterNumber(
            self.ANIM_FPS, self.tr('[Anim GIF] Frames per second'), 
            QgsProcessingParameterNumber.Integer, 20))

        self.addParameter(QgsProcessingParameterNumber(
            self.ANIM_SUBSAMPLE, self.tr('[Anim GIF] Number of agents to animate'), 
            QgsProcessingParameterNumber.Integer, 2000))

        self.addParameter(QgsProcessingParameterFileDestination(
            self.ANIM_GIF_PATH, self.tr('[Anim GIF] Output GIF file (optional)'), 
            'GIF files (*.gif)'))
        

    def processAlgorithm(self, parameters, context, feedback):
        if rasterio is None:
            raise QgsProcessingException('rasterio module not available. Please install rasterio in your QGIS environment.')

        boundary_src = self.parameterAsSource(parameters, self.INPUT_BOUNDARY, context)
        roads_src = self.parameterAsSource(parameters, self.INPUT_ROADS, context)
        worldpop_layer = self.parameterAsRasterLayer(parameters, self.INPUT_WORLDPOP, context)

        grid_size = self.parameterAsDouble(parameters, self.GRID_SIZE, context)
        min_pop = self.parameterAsDouble(parameters, self.MIN_POP, context)
        use_network = self.parameterAsBoolean(parameters, self.USE_NETWORK, context)
        agent_num = int(self.parameterAsInt(parameters, self.AGENT_NUM, context))
        lam = self.parameterAsDouble(parameters, self.LAMBDA, context)
        alpha = self.parameterAsDouble(parameters, self.ALPHA, context)
        beta = self.parameterAsDouble(parameters, self.BETA, context)
        gamma = self.parameterAsDouble(parameters, self.GAMMA, context)
        min_trips = self.parameterAsDouble(parameters, self.MIN_TRIPS, context)
        topn = int(self.parameterAsInt(parameters, self.TOPN_PER_ORIGIN, context))

        # --- Added for post-processing ---
        post_enable = self.parameterAsBoolean(parameters, self.POST_ENABLE, context)
        main_min_trips = self.parameterAsDouble(parameters, self.MAIN_MIN_TRIPS_EDGE, context)
        main_topk = self.parameterAsInt(parameters, self.MAIN_TOPK_EDGES, context)
        main_smooth = self.parameterAsDouble(parameters, self.MAIN_SMOOTH_TOL, context)
        main_buf = self.parameterAsDouble(parameters, self.MAIN_DISSOLVE_BUFFER, context)
        clus_eps = self.parameterAsDouble(parameters, self.CLUS_EPS_M, context)
        clus_minpts = self.parameterAsInt(parameters, self.CLUS_MINPTS, context)

        project_crs = QgsProject.instance().crs()

        hide_intra = self.parameterAsBoolean(parameters, self.HIDE_INTRA, context)

        anim_enable = self.parameterAsBoolean(parameters, self.ANIM_ENABLE, context)
        anim_duration = self.parameterAsDouble(parameters, self.ANIM_DURATION, context)
        anim_fps = self.parameterAsInt(parameters, self.ANIM_FPS, context)
        anim_subsample = self.parameterAsInt(parameters, self.ANIM_SUBSAMPLE, context)
        anim_gif_path = self.parameterAsFileOutput(parameters, self.ANIM_GIF_PATH, context)
        anim_on_road = self.parameterAsBoolean(parameters, self.ANIM_ON_ROAD, context)

        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        if seed is not None and int(seed) >= 0:
            np.random.seed(int(seed))

        # Merge boundary features
        feats = list(boundary_src.getFeatures())
        if not feats:
            raise QgsProcessingException('Boundary is empty.')
        geom = feats[0].geometry()
        for f in feats[1:]:
            geom = geom.combine(f.geometry())
        boundary_geom = geom
        extent = boundary_geom.boundingBox()
        xmin, xmax, ymin, ymax = extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum()

        # Prepare in-memory zone layer
        fields = QgsFields()
        fields.append(QgsField('zone_id', QVariant.Int))
        fields.append(QgsField('pop', QVariant.Double))
        fields.append(QgsField('cx', QVariant.Double))
        fields.append(QgsField('cy', QVariant.Double))
        # --- Added fields for generation/attraction (filled later) ---
        fields.append(QgsField('gen', QVariant.Double))
        fields.append(QgsField('att', QVariant.Double))
        fields.append(QgsField('net_ga', QVariant.Double))
        fields.append(QgsField('ratio_ga', QVariant.Double))

        zones_vl = QgsVectorLayer('Polygon?crs=' + project_crs.authid(), 'zones_mem', 'memory')
        prov = zones_vl.dataProvider()
        prov.addAttributes(fields)
        zones_vl.updateFields()

        # Iterate grid and sum WorldPop values per intersecting cell
        wp_path = worldpop_layer.source()
        created = 0
        zone_id = 0
        with rasterio.open(wp_path) as src:
            y = ymin
            while y < ymax:
                if feedback.isCanceled():
                    break
                x = xmin
                while x < xmax:
                    rect_coords = [
                        QgsPointXY(x, y),
                        QgsPointXY(x + grid_size, y),
                        QgsPointXY(x + grid_size, y + grid_size),
                        QgsPointXY(x, y + grid_size)
                    ]
                    cell = QgsGeometry.fromPolygonXY([rect_coords])
                    if not boundary_geom.intersects(cell):
                        x += grid_size
                        continue
                    poly_clip = boundary_geom.intersection(cell)
                    if poly_clip.isEmpty():
                        x += grid_size
                        continue
                    shapes = [json.loads(poly_clip.asJson())]
                    try:
                        out_img, _ = rio_mask(src, shapes, crop=True, all_touched=True)
                        data = out_img[0]
                        total_pop = float(np.nansum(data[np.isfinite(data)]))
                    except Exception:
                        total_pop = 0.0
                    if total_pop >= min_pop:
                        zone_id += 1
                        centroid = poly_clip.centroid().asPoint()
                        feat = QgsFeature(zones_vl.fields())
                        feat.setGeometry(poly_clip)
                        # gen/att/net/ratio diisi 0 sementara, akan dihitung setelah OD dibuat
                        feat.setAttributes([zone_id, total_pop, centroid.x(), centroid.y(), 0.0, 0.0, 0.0, 0.0])
                        prov.addFeature(feat)
                        created += 1
                    x += grid_size
                y += grid_size
        zones_vl.updateExtents()
        feedback.pushInfo(f'Zones created: {created}')
        if created < 2:
            raise QgsProcessingException('Fewer than two valid zones. Decrease grid size or lower the population threshold.')

        zones = list(zones_vl.getFeatures())
        pops = np.array([f['pop'] for f in zones], dtype=float)
        centroids = [QgsPointXY(f['cx'], f['cy']) for f in zones]
        n = len(zones)

        # Heuristic guard: large zone count with network distance can explode memory
        MAX_ZONES_NETWORK = 3000
        if use_network and n > MAX_ZONES_NETWORK:
            feedback.pushWarning(f'Zone count {n} exceeds safe network threshold {MAX_ZONES_NETWORK}. Falling back to geodesic distance.')
            use_network = False

        # Distance helpers
        def distance_matrix_euclid(points):
            N = len(points)
            D = np.zeros((N, N), dtype=float)
            is_geographic = project_crs.isGeographic()
            for i in range(N):
                pi = points[i]
                for j in range(i + 1, N):
                    pj = points[j]
                    if is_geographic:
                        d = haversine_m(pi, pj)
                    else:
                        d = math.hypot(pi.x() - pj.x(), pi.y() - pj.y())
                    D[i, j] = D[j, i] = d
            return D

        def distance_matrix_network(points):
            if nx is None:
                feedback.pushWarning('NetworkX not available. Falling back to Euclidean distance.')
                return distance_matrix_euclid(points)

            # Build road graph
            G = nx.Graph()
            for f in roads_src.getFeatures():
                geom = f.geometry()
                if geom.isMultipart():
                    lines = geom.asMultiPolyline()
                else:
                    lines = [geom.asPolyline()]
                for line in lines:
                    for a, b in zip(line[:-1], line[1:]):
                        pa = QgsPointXY(a)
                        pb = QgsPointXY(b)
                        w = haversine_m(pa, pb) if project_crs.isGeographic() else math.hypot(pa.x() - pb.x(), pa.y() - pb.y())
                        a_key = (round(pa.x(), 3), round(pa.y(), 3))
                        b_key = (round(pb.x(), 3), round(pb.y(), 3))
                        if a_key != b_key:
                            G.add_edge(a_key, b_key, weight=w)

            if G.number_of_edges() == 0:
                feedback.pushWarning('Road graph is empty. Falling back to Euclidean distance.')
                return distance_matrix_euclid(points)

            # Map centroids to nearest road nodes using a simple spatial index
            pts_layer = QgsVectorLayer('Point?crs=' + project_crs.authid(), 'nodes_mem', 'memory')
            pr = pts_layer.dataProvider()
            pr.addAttributes([QgsField('id', QVariant.Int)])
            pts_layer.updateFields()
            id_map = {}
            fid = 0
            for node in G.nodes:
                p = QgsPointXY(node[0], node[1])
                feat = QgsFeature(pts_layer.fields())
                feat.setGeometry(QgsGeometry.fromPointXY(p))
                feat.setAttributes([fid])
                pr.addFeature(feat)
                id_map[fid] = node
                fid += 1
            pts_layer.updateExtents()
            sindex = QgsSpatialIndex(pts_layer.getFeatures())

            nearest_nodes = []
            for p in points:
                nearest_fid = sindex.nearestNeighbor(QgsPointXY(p), 1)[0]
                node_key = id_map[nearest_fid]
                nearest_nodes.append(node_key)

            # Impedance-based cutoff to reduce search radius: exp(-lambda*d) < 1e-4
            cutoff = None
            if lam and lam > 0:
                cutoff = -math.log(1e-4) / lam

            N = len(points)
            D = np.zeros((N, N), dtype=float)
            # Cache Dijkstra results per unique source node
            uniq_nodes = list(set(nearest_nodes))
            dist_cache = {}
            for src_node in uniq_nodes:
                lengths = nx.single_source_dijkstra_path_length(G, src_node, cutoff=cutoff, weight='weight')
                dist_cache[src_node] = lengths

            for i in range(N):
                ni = nearest_nodes[i]
                for j in range(i + 1, N):
                    nj = nearest_nodes[j]
                    dij = dist_cache.get(ni, {}).get(nj)
                    if dij is None or not math.isfinite(dij):
                        # fallback
                        pi, pj = points[i], points[j]
                        dij = haversine_m(pi, pj) if project_crs.isGeographic() else math.hypot(pi.x() - pj.x(), pi.y() - pj.y())
                    D[i, j] = D[j, i] = dij
            return D

        D = distance_matrix_network(centroids) if use_network else distance_matrix_euclid(centroids)

        # Gravity model
        O = pops.copy()
        Dcap = pops.copy()
        O[O <= 0] = 1e-6
        Dcap[Dcap <= 0] = 1e-6

        if gamma and gamma > 0:
            F = 1.0 / np.power(np.maximum(D, 1.0), gamma)
        else:
            F = np.exp(-lam * D)
        Graw = np.outer(np.power(O, alpha), np.power(Dcap, beta)) * F

        total = Graw.sum()
        if total <= 0:
            raise QgsProcessingException('Raw OD matrix is zero. Adjust lambda/gamma or check population data.')
        scale = agent_num / total
        
        ODm = Graw * scale
        if hide_intra:
            np.fill_diagonal(ODm, 0.0)

            
        # Build keep mask: Top-N per origin and min_trips
        if topn > 0 and topn < n:
            keep_mask = np.zeros_like(ODm, dtype=bool)
            for i in range(n):
                idx = np.argpartition(ODm[i, :], -topn)[-topn:]
                keep_mask[i, idx] = True
        else:
            keep_mask = np.ones_like(ODm, dtype=bool)

        # Write OD CSV (filtered)
        od_csv_path = self.parameterAsFileOutput(parameters, self.OUTPUT_OD, context)
        with open(od_csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['i', 'j', 'trips'])
            for i in range(n):
                zi = int(zones[i]['zone_id'])
                for j in range(n):
                    val = float(ODm[i, j])
                    if not keep_mask[i, j]:
                        continue
                    if val < min_trips:
                        continue
                    w.writerow([zi, int(zones[j]['zone_id']), val])

        # Build weighted OD line layer
        edges_fields = QgsFields()
        edges_fields.append(QgsField('i', QVariant.Int))
        edges_fields.append(QgsField('j', QVariant.Int))
        edges_fields.append(QgsField('trips', QVariant.Double))
        edges_vl = QgsVectorLayer('LineString?crs=' + project_crs.authid(), 'od_edges_mem', 'memory')
        eprov = edges_vl.dataProvider()
        eprov.addAttributes(edges_fields)
        edges_vl.updateFields()
        for i in range(n):
            pi = centroids[i]
            zi = int(zones[i]['zone_id'])
            for j in range(n):
                val = float(ODm[i, j])
                if not keep_mask[i, j]:
                    continue
                if val <= 0 or val < min_trips:
                    continue
                pj = centroids[j]
                geom = QgsGeometry.fromPolylineXY([pi, pj])
                feat = QgsFeature(edges_vl.fields())
                feat.setGeometry(geom)
                feat.setAttributes([zi, int(zones[j]['zone_id']), val])
                eprov.addFeature(feat)
        edges_vl.updateExtents()

        # Sample individuals according to OD probabilities
        probs = ODm / ODm.sum()
        flat = probs.ravel()
        idx = np.random.choice(np.arange(n * n), size=agent_num, p=flat)
        oi = idx // n
        dj = idx % n

        def random_point_in_geom(qgs_geom: QgsGeometry):
            bbox = qgs_geom.boundingBox()
            tries = 0
            while True:
                x = np.random.uniform(bbox.xMinimum(), bbox.xMaximum())
                y = np.random.uniform(bbox.yMinimum(), bbox.yMaximum())
                p = QgsPointXY(x, y)
                if qgs_geom.contains(QgsGeometry.fromPointXY(p)):
                    return p
                tries += 1
                if tries > 1000:
                    return qgs_geom.centroid().asPoint()

        persons_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_PERSONS, context)
        with open(persons_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['orig_x', 'orig_y', 'dest_x', 'dest_y', 'zone_o', 'zone_d'])
            for k in range(agent_num):
                zo = zones[oi[k]]
                zd = zones[dj[k]]
                po = random_point_in_geom(zo.geometry())
                pd = random_point_in_geom(zd.geometry())
                w.writerow([po.x(), po.y(), pd.x(), pd.y(), int(zo['zone_id']), int(zd['zone_id'])])

        # --- Post-processing: Generation & Attraction per zona (ditulis ke layer zones) ---
        gen_sum = {int(f['zone_id']): 0.0 for f in zones}
        att_sum = {int(f['zone_id']): 0.0 for f in zones}
        for i in range(n):
            zi = int(zones[i]['zone_id'])
            row = ODm[i, :]
            gen_sum[zi] += float(row.sum())
        for j in range(n):
            zj = int(zones[j]['zone_id'])
            col = ODm[:, j]
            att_sum[zj] += float(col.sum())

        with edit(zones_vl):
            for f in zones_vl.getFeatures():
                zid = int(f['zone_id'])
                g = float(gen_sum.get(zid, 0.0))
                a = float(att_sum.get(zid, 0.0))
                net_ga = g - a
                ratio_ga = (g / a) if a > 0 else None
                zones_vl.changeAttributeValue(f.id(), zones_vl.fields().indexFromName('gen'), g)
                zones_vl.changeAttributeValue(f.id(), zones_vl.fields().indexFromName('att'), a)
                zones_vl.changeAttributeValue(f.id(), zones_vl.fields().indexFromName('net_ga'), net_ga)
                zones_vl.changeAttributeValue(f.id(), zones_vl.fields().indexFromName('ratio_ga'), ratio_ga if ratio_ga is not None else 0.0)

        # Write vector outputs using V3 API with backward/forward compatible unpacking
        zones_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_ZONES, context)
        edges_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_EDGES, context)

        def _unpack_writer_result(result):
            if isinstance(result, tuple):
                if len(result) >= 3:
                    return result[0], result[1], result[2]
            return result, '', ''

        save_opts = QgsVectorFileWriter.SaveVectorOptions()
        if zones_path and zones_path.lower().endswith('.gpkg'):
            save_opts.driverName = 'GPKG'
        elif zones_path and zones_path.lower().endswith('.shp'):
            save_opts.driverName = 'ESRI Shapefile'
        save_opts.layerName = 'zones'

        res_tuple1 = QgsVectorFileWriter.writeAsVectorFormatV3(
            zones_vl, zones_path, QgsProject.instance().transformContext(), save_opts
        )
        res1, err1, new1 = _unpack_writer_result(res_tuple1)
        if res1 != QgsVectorFileWriter.NoError:
            raise QgsProcessingException(f'Failed writing zones: {err1}')

        save_opts2 = QgsVectorFileWriter.SaveVectorOptions()
        if edges_path and edges_path.lower().endswith('.gpkg'):
            save_opts2.driverName = 'GPKG'
        elif edges_path and edges_path.lower().endswith('.shp'):
            save_opts2.driverName = 'ESRI Shapefile'
        save_opts2.layerName = 'od_edges'

        res_tuple2 = QgsVectorFileWriter.writeAsVectorFormatV3(
            edges_vl, edges_path, QgsProject.instance().transformContext(), save_opts2
        )
        res2, err2, new2 = _unpack_writer_result(res_tuple2)
        if res2 != QgsVectorFileWriter.NoError:
            raise QgsProcessingException(f'Failed writing edges: {err2}')

        # --- Post-processing lanjutan (opsional) ---
        outputs = {
            self.OUTPUT_ZONES: zones_path,
            self.OUTPUT_OD: od_csv_path,
            self.OUTPUT_EDGES: edges_path,
            self.OUTPUT_PERSONS: persons_csv
        }

        if not post_enable:
            return outputs

        # Pastikan modul processing tersedia
        try:
            import processing
        except Exception:
            feedback.pushWarning('QGIS processing module unavailable. Skipping post-processing.')
            return outputs

        # --- Main Transit Routes sebagai LINE (tanpa orderby/sort processing) ---
        tmp_edges = edges_vl

        # 1) Filter edges trip >= ambang
        try:
            flt = processing.run(
                'native:extractbyattribute',
                {'INPUT': tmp_edges, 'FIELD': 'trips', 'OPERATOR': 3, 'VALUE': main_min_trips, 'OUTPUT': 'memory:'},
                context=context, feedback=feedback
            )['OUTPUT']
        except Exception:
            flt = tmp_edges  # fallback: pakai semua

        # 2) Ambil semua fitur ke list dan sort di Python
        feat_list = list(flt.getFeatures())
        feat_list.sort(key=lambda f: float(f['trips']) if f['trips'] is not None else 0.0, reverse=True)
        top_feats = feat_list[:max(main_topk, 1)]

        # 3) Buat layer garis "main_routes" lalu isi Top-K
        main_routes = QgsVectorLayer('LineString?crs=' + project_crs.authid(), 'main_routes_mem', 'memory')
        mr_dp = main_routes.dataProvider()
        # gunakan field yang sama dengan edges: i, j, trips
        mr_dp.addAttributes(edges_vl.fields())
        main_routes.updateFields()

        new_feats = []
        for f in top_feats:
            nf = QgsFeature(main_routes.fields())
            nf.setGeometry(f.geometry())
            nf.setAttributes([int(f['i']), int(f['j']), float(f['trips'])])
            new_feats.append(nf)
        mr_dp.addFeatures(new_feats)
        main_routes.updateExtents()

        # 4) Smooth opsional (tetap LineString)
        if main_smooth and main_smooth > 0:
            try:
                main_routes = processing.run(
                    'native:smoothgeometry',
                    {'INPUT': main_routes, 'ITERATIONS': 1, 'OFFSET': 0, 'MAX_ANGLE': 180,
                     'MAX_DISTANCE': main_smooth, 'OUTPUT': 'memory:'},
                    context=context, feedback=feedback
                )['OUTPUT']
            except Exception:
                pass  # jika gagal, pakai tanpa smoothing

        # 5) Simpan sebagai OUTPUT_MAIN_ROUTES (LineString)
        main_routes_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_MAIN_ROUTES, context)
        save_opts3 = QgsVectorFileWriter.SaveVectorOptions()
        if main_routes_path and main_routes_path.lower().endswith('.gpkg'):
            save_opts3.driverName = 'GPKG'
        elif main_routes_path and main_routes_path.lower().endswith('.shp'):
            save_opts3.driverName = 'ESRI Shapefile'
        save_opts3.layerName = 'main_routes'

        res_tuple3 = QgsVectorFileWriter.writeAsVectorFormatV3(
            main_routes, main_routes_path, QgsProject.instance().transformContext(), save_opts3
        )
        res3, err3, new3 = _unpack_writer_result(res_tuple3)
        if res3 != QgsVectorFileWriter.NoError:
            feedback.pushWarning(f'Failed writing main routes: {err3}')
        else:
            outputs[self.OUTPUT_MAIN_ROUTES] = main_routes_path


        # 2) DBSCAN Clustering: origin, destination, dan midpoint edges
        #    Buat layer titik origin/destination dari persons CSV
        def _csv_to_points(csv_path, role):
            # Buat layer memory titik
            vl = QgsVectorLayer(f'Point?crs={project_crs.authid()}', f'{role}_pts', 'memory')
            if not vl.isValid():
                raise QgsProcessingException(f'Failed to create memory layer for {role} points.')

            # Tambah skema atribut secepatnya, jangan pegang provider lama-lama
            if not vl.dataProvider().addAttributes([
                QgsField('zone_o', QVariant.Int),
                QgsField('zone_d', QVariant.Int),
                QgsField('role', QVariant.String)
            ]):
                raise QgsProcessingException(f'Failed to add fields for {role} points.')
            vl.updateFields()

            # Kumpulkan fitur lalu sekali addFeatures
            feats = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    try:
                        if role == 'origin':
                            x, y = float(r['orig_x']), float(r['orig_y'])
                        else:
                            x, y = float(r['dest_x']), float(r['dest_y'])
                        g = QgsGeometry.fromPointXY(QgsPointXY(x, y))
                        feat = QgsFeature(vl.fields())
                        feat.setGeometry(g)
                        feat.setAttributes([int(r['zone_o']), int(r['zone_d']), role[0]])
                        feats.append(feat)
                    except Exception:
                        # Lewati baris yang rusak agar provider tidak error
                        continue

            if feats:
                ok = vl.dataProvider().addFeatures(feats)
                if not ok:
                    raise QgsProcessingException(f'Failed to add features for {role} points.')
                vl.updateExtents()

            return vl

        origin_pts = _csv_to_points(persons_csv, 'origin')
        dest_pts   = _csv_to_points(persons_csv, 'destination')

        # Midpoint dari edges
        mid_fields = QgsFields()
        mid_fields.append(QgsField('i', QVariant.Int))
        mid_fields.append(QgsField('j', QVariant.Int))
        mid_fields.append(QgsField('trips', QVariant.Double))
        mid_pts = QgsVectorLayer('Point?crs=' + project_crs.authid(), 'od_midpoints', 'memory')
        mpdp = mid_pts.dataProvider()
        mpdp.addAttributes(mid_fields)
        mid_pts.updateFields()
        mf = []
        for f in edges_vl.getFeatures():
            g = f.geometry()
            c = g.centroid().asPoint()
            feat = QgsFeature(mid_pts.fields())
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(c.x(), c.y())))
            feat.setAttributes([int(f['i']), int(f['j']), float(f['trips'])])
            mf.append(feat)
        mpdp.addFeatures(mf); mid_pts.updateExtents()

        # Jalankan DBSCAN
        def _run_dbscan(vlayer, eps_m, minpts):
            try:
                result = processing.run(
                    'native:dbscanclustering',
                    {'INPUT': vlayer, 'EPS': eps_m, 'MIN_POINTS': int(minpts),
                     'FIELD_NAME': 'cluster_id', 'OUTPUT': 'memory:'},
                    context=context, feedback=feedback
                )
                return result['OUTPUT']
            except Exception as e:
                feedback.pushWarning(f'DBSCAN failed: {e}')
                return vlayer  # fallback tanpa cluster_id
        origin_clusters = _run_dbscan(origin_pts, clus_eps, clus_minpts)
        dest_clusters = _run_dbscan(dest_pts, clus_eps, clus_minpts)
        mid_clusters = _run_dbscan(mid_pts, clus_eps, clus_minpts)

        # Hull per cluster
        def _hulls(points_layer, eps_for_buffer):
            """
            Polygon "hull" per cluster_id:
            - n>=3  : convex hull MultiPoint
            - n==2  : union buffer tipis dari 2 titik
            - n==1  : buffer kecil dari 1 titik
            Noise (-1) dan NULL di-skip.
            """
            # Pastikan field cluster_id ada
            if points_layer.fields().indexFromName('cluster_id') == -1:
                return None

            out_vl = QgsVectorLayer('Polygon?crs=' + points_layer.crs().authid(),
                                    points_layer.name() + '_hulls', 'memory')
            dp = out_vl.dataProvider()
            dp.addAttributes([QgsField('cluster_id', QVariant.Int),
                              QgsField('n_points', QVariant.Int)])
            out_vl.updateFields()

            # Kelompokkan per cluster (skip noise dan NULL)
            clusters = {}
            for f in points_layer.getFeatures():
                cid = f['cluster_id']
                if cid is None:
                    continue               # NULL → skip
                try:
                    cid = int(cid)
                except Exception:
                    continue               # tak bisa cast → skip
                if cid < 0:
                    continue               # noise → skip
                clusters.setdefault(cid, []).append(f)

            if not clusters:
                return None

            buf_r = max(1.0, float(eps_for_buffer) * 0.25)  # radius buffer kecil

            new_polys = []
            for cid, feats in clusters.items():
                n = len(feats)
                if n >= 3:
                    geoms = [ff.geometry() for ff in feats]
                    multi = QgsGeometry.unaryUnion(geoms)
                    hull = multi.convexHull()
                    poly = hull
                elif n == 2:
                    g1 = feats[0].geometry().buffer(buf_r, 8)
                    g2 = feats[1].geometry().buffer(buf_r, 8)
                    poly = g1.combine(g2)
                elif n == 1:
                    poly = feats[0].geometry().buffer(buf_r, 16)
                else:
                    continue

                nf = QgsFeature(out_vl.fields())
                nf.setGeometry(poly)
                nf.setAttributes([cid, n])
                new_polys.append(nf)

            if new_polys:
                dp.addFeatures(new_polys)
                out_vl.updateExtents()
                return out_vl
            return None

        origin_hulls = _hulls(origin_clusters, clus_eps)
        dest_hulls   = _hulls(dest_clusters, clus_eps)
        mid_hulls    = _hulls(mid_clusters,   clus_eps)


        # Simpan semua klaster dan hull
        def _save_vec(vl, path, lname, key_name):
            if not vl or not path:
                return
            so = QgsVectorFileWriter.SaveVectorOptions()
            if path.lower().endswith('.gpkg'):
                so.driverName = 'GPKG'
            elif path.lower().endswith('.shp'):
                so.driverName = 'ESRI Shapefile'
            so.layerName = lname
            res_t = QgsVectorFileWriter.writeAsVectorFormatV3(
                vl, path, QgsProject.instance().transformContext(), so
            )
            r, e, n = _unpack_writer_result(res_t)
            if r != QgsVectorFileWriter.NoError:
                feedback.pushWarning(f'Failed writing {lname}: {e}')
            else:
                outputs[key_name] = path

        # Panggilan
        _save_vec(origin_clusters, self.parameterAsOutputLayer(parameters, self.OUTPUT_ORIGIN_CLUSTERS, context), 'origin_clusters', self.OUTPUT_ORIGIN_CLUSTERS)
        _save_vec(dest_clusters,   self.parameterAsOutputLayer(parameters, self.OUTPUT_DEST_CLUSTERS, context),   'dest_clusters',   self.OUTPUT_DEST_CLUSTERS)
        _save_vec(mid_clusters,    self.parameterAsOutputLayer(parameters, self.OUTPUT_ODMID_CLUSTERS, context),  'odmid_clusters',  self.OUTPUT_ODMID_CLUSTERS)
        if origin_hulls:
            _save_vec(origin_hulls, self.parameterAsOutputLayer(parameters, self.OUTPUT_ORIGIN_HULLS, context), 'origin_hulls', self.OUTPUT_ORIGIN_HULLS)
        if dest_hulls:
            _save_vec(dest_hulls, self.parameterAsOutputLayer(parameters, self.OUTPUT_DEST_HULLS, context), 'dest_hulls', self.OUTPUT_DEST_HULLS)
        if mid_hulls:
            _save_vec(mid_hulls, self.parameterAsOutputLayer(parameters, self.OUTPUT_ODMID_HULLS, context), 'odmid_hulls', self.OUTPUT_ODMID_HULLS)

        # ===================== Render GIF Animation =====================
        if anim_enable and anim_gif_path:
            try:
                try:
                    import imageio.v3 as iio
                except Exception:
                    import imageio as iio
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from matplotlib.collections import LineCollection, PatchCollection
                from matplotlib.patches import Polygon as MplPolygon
                
                # Baca opsi follow roads
                use_on_road = bool(anim_on_road)
                
                # Read persons and subsample agents
                rows = []
                with open(persons_csv, 'r', encoding='utf-8') as f:
                    rdr = list(csv.DictReader(f))
                total_agents = len(rdr)
                if total_agents == 0:
                    feedback.pushWarning('No agents found in persons CSV. Skipping GIF rendering.')
                    raise RuntimeError('No agents')

                if total_agents > anim_subsample:
                    idx_choice = np.random.choice(np.arange(total_agents), size=anim_subsample, replace=False)
                    rows = [rdr[i] for i in idx_choice]
                else:
                    rows = rdr

                # Compute bounds from O/D
                xs = [float(r['orig_x']) for r in rows] + [float(r['dest_x']) for r in rows]
                ys = [float(r['orig_y']) for r in rows] + [float(r['dest_y']) for r in rows]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

                # ================= Background geometry: roads and boundary =================
                roads_lines = []
                bg_xmin = bg_ymin = float('inf')
                bg_xmax = bg_ymax = float('-inf')

                for f in roads_src.getFeatures():
                    g = f.geometry()
                    if g is None or g.isEmpty():
                        continue
                    if g.isMultipart():
                        lines = g.asMultiPolyline()
                    else:
                        lines = [g.asPolyline()]
                    for line in lines:
                        if len(line) >= 2:
                            arr = np.array([[pt.x(), pt.y()] for pt in line], dtype=float)
                            roads_lines.append(arr)
                            bg_xmin = min(bg_xmin, arr[:, 0].min()); bg_xmax = max(bg_xmax, arr[:, 0].max())
                            bg_ymin = min(bg_ymin, arr[:, 1].min()); bg_ymax = max(bg_ymax, arr[:, 1].max())

                boundary_patches = []
                for f in boundary_src.getFeatures():
                    g = f.geometry()
                    if g is None or g.isEmpty():
                        continue
                    if g.isMultipart():
                        polys = g.asMultiPolygon()
                        poly_list = [ring for poly in polys for ring in ([poly[0]] if poly else [])]
                    else:
                        poly = g.asPolygon()
                        poly_list = [poly[0]] if poly else []
                    for ring in poly_list:
                        if not ring:
                            continue
                        arr = np.array([[pt.x(), pt.y()] for pt in ring], dtype=float)
                        boundary_patches.append(MplPolygon(arr, closed=True, fill=False))
                        bg_xmin = min(bg_xmin, arr[:, 0].min()); bg_xmax = max(bg_xmax, arr[:, 0].max())
                        bg_ymin = min(bg_ymin, arr[:, 1].min()); bg_ymax = max(bg_ymax, arr[:, 1].max())

                # Expand scene bounds to include background
                if not np.isinf(bg_xmin):
                    xmin = min(xmin, bg_xmin); xmax = max(xmax, bg_xmax)
                    ymin = min(ymin, bg_ymin); ymax = max(ymax, bg_ymax)

                mx = (xmax - xmin) * 0.05 or 1.0
                my = (ymax - ymin) * 0.05 or 1.0
                xmin -= mx; xmax += mx; ymin -= my; ymax += my

                # ================= Paths on road network (optional) =================
                use_on_road = bool(anim_on_road)

                agent_paths = None
                if use_on_road:
                    if nx is None:
                        feedback.pushWarning('NetworkX not available. Falling back to straight-line animation.')
                        use_on_road = False
                    else:
                        # Build a light-weight road graph for animation
                        G = nx.Graph()
                        for f in roads_src.getFeatures():
                            g = f.geometry()
                            if g is None or g.isEmpty():
                                continue
                            if g.isMultipart():
                                lines = g.asMultiPolyline()
                            else:
                                lines = [g.asPolyline()]
                            for line in lines:
                                for a, b in zip(line[:-1], line[1:]):
                                    pa = (a.x(), a.y()); pb = (b.x(), b.y())
                                    if pa == pb:
                                        continue
                                    w = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                                    G.add_edge(pa, pb, weight=w)

                        if G.number_of_edges() == 0:
                            feedback.pushWarning('Road graph empty. Falling back to straight-line animation.')
                            use_on_road = False
                        else:
                            # Make spatial index of nodes for nearest snapping
                            pts_layer = QgsVectorLayer('Point?crs=' + project_crs.authid(), 'nodes_mem', 'memory')
                            pr = pts_layer.dataProvider()
                            pr.addAttributes([QgsField('id', QVariant.Int)])
                            pts_layer.updateFields()
                            id_map = {}
                            fid = 0
                            for node in G.nodes:
                                feat = QgsFeature(pts_layer.fields())
                                feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(node[0], node[1])))
                                feat.setAttributes([fid])
                                pr.addFeature(feat)
                                id_map[fid] = node
                                fid += 1
                            pts_layer.updateExtents()
                            sindex = QgsSpatialIndex(pts_layer.getFeatures())

                            def nearest_node_xy(x, y):
                                try:
                                    nf = sindex.nearestNeighbor(QgsPointXY(x, y), 1)[0]
                                    return id_map[nf]
                                except Exception:
                                    return None

                            # Precompute polyline path and cumulative lengths per agent
                            agent_paths = []
                            for r in rows:
                                ox, oy = float(r['orig_x']), float(r['orig_y'])
                                dx, dy = float(r['dest_x']), float(r['dest_y'])
                                ni = nearest_node_xy(ox, oy)
                                nj = nearest_node_xy(dx, dy)
                                coords = None
                                if ni is not None and nj is not None:
                                    try:
                                        node_path = nx.shortest_path(G, ni, nj, weight='weight')
                                        coords = np.array([[p[0], p[1]] for p in node_path], dtype=float)
                                    except Exception:
                                        coords = None
                                if coords is None or coords.shape[0] < 2:
                                    # fall back to straight segment if no route
                                    coords = np.array([[ox, oy], [dx, dy]], dtype=float)

                                seg = coords[1:] - coords[:-1]
                                seglen = np.hypot(seg[:, 0], seg[:, 1])
                                cum = np.concatenate(([0.0], np.cumsum(seglen))) if seglen.size else np.array([0.0, 1.0])
                                total = float(cum[-1]) if cum.size else 1.0
                                agent_paths.append((coords, cum, total))

                frames = int(anim_duration * anim_fps)
                frames = max(frames, 1)

                frames_buf = []
                for fidx in range(frames + 1):
                    u = fidx / float(frames)  # 0..1
                    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)

                    # --- background ---
                    if roads_lines:
                        lc = LineCollection(roads_lines, linewidths=0.4, colors='0.5', alpha=0.3, zorder=0)
                        ax.add_collection(lc)
                    if boundary_patches:
                        pc = PatchCollection(boundary_patches, match_original=False,
                                             facecolor='none', edgecolor='r', linewidths=0.6, alpha=0.3, zorder=1)
                        ax.add_collection(pc)

                    # --- agents ---
                    if use_on_road and agent_paths is not None:
                        X = []
                        Y = []
                        for coords, cum, total in agent_paths:
                            if total <= 0:
                                X.append(coords[-1, 0]); Y.append(coords[-1, 1]); continue
                            dist = u * total
                            # seg index: rightmost cum[k] <= dist
                            k = int(np.searchsorted(cum, dist, side='right') - 1)
                            if k >= len(coords) - 1:
                                x, y = coords[-1, 0], coords[-1, 1]
                            else:
                                d0 = cum[k]; d1 = cum[k + 1]
                                t = 0.0 if d1 <= d0 else (dist - d0) / (d1 - d0)
                                p0 = coords[k]; p1 = coords[k + 1]
                                x = p0[0] + t * (p1[0] - p0[0])
                                y = p0[1] + t * (p1[1] - p0[1])
                            X.append(x); Y.append(y)
                        ax.scatter(X, Y, s=2, zorder=2)
                    else:
                        # straight-line interpolation (default behaviour)
                        X = []
                        Y = []
                        for r in rows:
                            ox, oy = float(r['orig_x']), float(r['orig_y'])
                            dx, dy = float(r['dest_x']), float(r['dest_y'])
                            X.append(ox + u * (dx - ox))
                            Y.append(oy + u * (dy - oy))
                        ax.scatter(X, Y, s=2, zorder=2)

                    ax.set_axis_off()
                    fig.tight_layout(pad=0)

                    # Canvas → RGB
                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    frame = buf.reshape((h, w, 4))[:, :, :3]
                    frames_buf.append(frame)
                    plt.close(fig)

                iio.imwrite(anim_gif_path, frames_buf, duration=1.0 / anim_fps, loop=0)
                outputs[self.ANIM_GIF_PATH] = anim_gif_path
                feedback.pushInfo(f'GIF animation saved: {anim_gif_path}')
            except Exception as e:
                feedback.pushWarning(f'Failed to render GIF animation: {e}')
        # ================== End Option B: Direct GIF Animation Rendering ==================

        return outputs

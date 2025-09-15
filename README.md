# AutoOD & AgentSim from WorldPop

**AutoOD & AgentSim from WorldPop** is a QGIS plugin for building **Origin–Destination (OD) matrices** and **simulating individual trips** using **WorldPop raster data**.  
It supports mobility analysis, transport planning, and spatial interaction studies in the context of **Urban and Regional Planning**.  

---

## ✨ Key Features
1. **Zone Grid Builder**  
   - Creates zone grids from boundary polygons.  
   - Aggregates WorldPop population within each zone.  

2. **Gravity Model OD Matrix**  
   - Generates OD matrix based on a gravity model.  
   - Configurable parameters: α, β, λ, γ.  

3. **Agent-Based Simulation**  
   - Samples individual trips from the OD matrix.  
   - Output: CSV of origin–destination coordinates per agent.  

4. **Post-Processing (optional)**  
   - Calculates trip *generation* & *attraction* per zone.  
   - Extracts *main routes* from weighted OD lines.  
   - DBSCAN clustering (origins, destinations, and midpoints) with hull polygons.  

5. **GIF Animation (optional)**  
   - Visualizes trip movements from origin to destination.  
   - Displays boundary (red, faded) and road network (gray, faded) as background.  
   - *Follow Roads* option lets agents move along the road network (shortest path).  

---

## 📥 Installation
1. Clone this repository into your QGIS plugins folder:  
   ```bash
   git clone https://github.com/firmanaf/AutoOD-AgentSim-WorldPop.git

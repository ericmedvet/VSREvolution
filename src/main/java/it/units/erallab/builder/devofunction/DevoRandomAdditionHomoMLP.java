package it.units.erallab.builder.devofunction;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;

import java.util.Objects;

public class DevoRandomAdditionHomoMLP extends DevoHomoMLP {

  public DevoRandomAdditionHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, int nInitial, int nStep) {
    super(innerLayerRatio, nOfInnerLayers, signals, nInitial, nStep);
  }

  @Override
  protected Grid<? extends SensingVoxel> createBody(Robot<? extends SensingVoxel> previous, Grid<Double> strengths, SensingVoxel voxelPrototype) {
    Grid<Double> selected;
    if (previous == null) {
      selected = Utils.gridConnected(strengths, Double::compareTo, nInitial);
    } else {
      Grid<Double> start = Grid.create(previous.getVoxels(), v -> v == null ? Math.random() : -2d);
      int n = (int) previous.getVoxels().values().stream().filter(Objects::nonNull).count() + nStep;
      selected = Utils.gridConnected(start, Double::compareTo, n);
    }
    Grid<SensingVoxel> body = Grid.create(selected, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }

}

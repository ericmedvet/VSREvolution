package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import it.units.malelab.jgea.core.util.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

public class DevoConditionedHomoMLP extends DevoHomoMLP implements PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot<? extends SensingVoxel>>> {

  private final Function<Voxel, Double> selectionFunction;
  private final boolean maxFirst;

  public DevoConditionedHomoMLP(double innerLayerRatio, int nOfInnerLayers, int signals, Function<Voxel, Double> selectionFunction, boolean maxFirst, int nInitial, int nStep) {
    super(innerLayerRatio, nOfInnerLayers, signals, nInitial, nStep);
    this.selectionFunction = selectionFunction;
    this.maxFirst = maxFirst;
  }

  public DevoConditionedHomoMLP(
      double innerLayerRatio, int nOfInnerLayers, int signals,
      Function<Voxel, Double> selectionFunction,
      int nInitial, int nStep) {
    this(innerLayerRatio, nOfInnerLayers, signals, selectionFunction, false, nInitial, nStep);
  }

  @Override
  protected Grid<? extends SensingVoxel> createBody(Robot<? extends SensingVoxel> previous, Grid<Double> strengths, SensingVoxel voxelPrototype) {
    Grid<SensingVoxel> body;
    if (previous == null) {
      Grid<Double> selected = Utils.gridConnected(strengths, Double::compareTo, nInitial);
      body = Grid.create(selected, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
    } else {
      // sort voxels according to the given function
      Grid<? extends SensingVoxel> previousBody = previous.getVoxels();
      List<Grid.Entry<? extends SensingVoxel>> sortedVoxels = previousBody.stream()
          .filter(e -> e.getValue() != null)
          .sorted(Comparator.comparing(e -> selectionFunction.apply(e.getValue())))
          .collect(Collectors.toList());
      if (maxFirst) {
        Collections.reverse(sortedVoxels);
      }
      List<Pair<Integer, Integer>> nextPositions = sortedVoxels.stream()
          .map(e -> getStrengthSortedEmptyNeighborsPositions(previousBody, e, strengths))
          .flatMap(List::stream).collect(Collectors.toList());
      body = Grid.create(previousBody, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
      for (int i = 0; i < nStep; i++) {
        body.set(nextPositions.get(i).first(), nextPositions.get(i).second(), SerializationUtils.clone(voxelPrototype));
      }
    }
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }

  private static <V> List<Pair<Integer, Integer>> getEmptyNeighborsPositions(Grid<? extends V> grid, Grid.Entry<? extends V> entry) {
    List<Pair<Integer, Integer>> emptyNeighbors = new ArrayList<>();
    for (int d : new int[]{-1, 1}) {
      int x = entry.getX();
      int y = entry.getY();
      if ((x + d) >= 0 && (x + d) < grid.getW() && grid.get(x + d, y) == null) {
        emptyNeighbors.add(Pair.of(x + d, y));
      }
      if ((y + d) >= 0 && (y + d) < grid.getH() && grid.get(x, y + d) == null) {
        emptyNeighbors.add(Pair.of(x, y + d));
      }
    }
    return emptyNeighbors;
  }

  private static <V> List<Pair<Integer, Integer>> getStrengthSortedEmptyNeighborsPositions(Grid<? extends V> grid, Grid.Entry<? extends V> entry, Grid<Double> strengths) {
    List<Pair<Integer, Integer>> emptyNeighbors = getEmptyNeighborsPositions(grid, entry);
    if (emptyNeighbors.isEmpty()) {
      return emptyNeighbors;
    }
    return strengths.stream()
        .sorted(Comparator.comparingDouble(Grid.Entry::getValue))
        .map(e -> Pair.of(e.getX(), e.getY()))
        .filter(emptyNeighbors::contains)
        .collect(Collectors.toList());
  }

}

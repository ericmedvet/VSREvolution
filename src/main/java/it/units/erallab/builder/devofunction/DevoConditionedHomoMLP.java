package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;

import java.util.*;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

public class DevoConditionedHomoMLP extends DevoHomoMLP implements PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot>> {

  private final Function<Voxel, Double> selectionFunction;
  private final boolean maxFirst;

  public DevoConditionedHomoMLP(
      double innerLayerRatio,
      int nOfInnerLayers,
      int signals,
      Function<Voxel, Double> selectionFunction,
      boolean maxFirst,
      int nInitial,
      int nStep
  ) {
    super(innerLayerRatio, nOfInnerLayers, signals, nInitial, nStep, 0d);
    this.selectionFunction = selectionFunction;
    this.maxFirst = maxFirst;
  }

  public DevoConditionedHomoMLP(
      double innerLayerRatio, int nOfInnerLayers, int signals,
      Function<Voxel, Double> selectionFunction,
      int nInitial, int nStep
  ) {
    this(innerLayerRatio, nOfInnerLayers, signals, selectionFunction, false, nInitial, nStep);
  }

  @Override
  protected Grid<Voxel> createBody(Robot previous, Grid<Double> strengths, Voxel voxelPrototype) {
    Grid<Voxel> body;
    if (previous == null) {
      Grid<Double> selected = Utils.gridConnected(strengths, Double::compareTo, nInitial);
      body = Grid.create(selected, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
    } else {
      // sort voxels according to the given function
      Grid<Voxel> previousBody = previous.getVoxels();
      List<Grid.Entry<Voxel>> sortedVoxels = previousBody.stream()
          .filter(e -> e.value() != null)
          .sorted(Comparator.comparing(e -> selectionFunction.apply(e.value())))
          .collect(Collectors.toList());
      if (maxFirst) {
        Collections.reverse(sortedVoxels);
      }
      List<Grid.Key> nextPositions = sortedVoxels.stream()
          .map(e -> getStrengthSortedEmptyNeighborsPositions(previousBody, e, strengths))
          .flatMap(List::stream).toList();
      body = Grid.create(previousBody, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
      for (int i = 0; i < nStep; i++) {
        body.set(nextPositions.get(i).x(), nextPositions.get(i).y(), SerializationUtils.clone(voxelPrototype));
      }
    }
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }

  private static List<Grid.Key> getEmptyNeighborsPositions(Grid<Voxel> grid, Grid.Entry<Voxel> entry) {
    List<Grid.Key> emptyNeighbors = new ArrayList<>();
    for (int d : new int[]{-1, 1}) {
      int x = entry.key().x();
      int y = entry.key().y();
      if ((x + d) >= 0 && (x + d) < grid.getW() && grid.get(x + d, y) == null) {
        emptyNeighbors.add(new Grid.Key(x + d, y));
      }
      if ((y + d) >= 0 && (y + d) < grid.getH() && grid.get(x, y + d) == null) {
        emptyNeighbors.add(new Grid.Key(x, y + d));
      }
    }
    return emptyNeighbors;
  }

  private static List<Grid.Key> getStrengthSortedEmptyNeighborsPositions(
      Grid<Voxel> grid,
      Grid.Entry<Voxel> entry,
      Grid<Double> strengths
  ) {
    List<Grid.Key> emptyNeighbors = getEmptyNeighborsPositions(grid, entry);
    if (emptyNeighbors.isEmpty()) {
      return emptyNeighbors;
    }
    return strengths.stream()
        .sorted(Comparator.comparingDouble(Grid.Entry::value))
        .map(Grid.Entry::key)
        .filter(emptyNeighbors::contains)
        .toList();
  }

}

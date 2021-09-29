package it.units.erallab.devo;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.robot.BodyAndSinusoidal;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;
import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoFixedPhasesValues implements PrototypedFunctionBuilder<Grid<double[]>, UnaryOperator<Robot<?>>> {

  private final double frequency;
  private final double amplitude;
  private final int nStep;

  public DevoFixedPhasesValues(double frequency, double amplitude, int nStep) {
    this.frequency = frequency;
    this.amplitude = amplitude;
    this.nStep = nStep;
  }

  @Override
  public Function<Grid<double[]>, UnaryOperator<Robot<?>>> buildFor(UnaryOperator<Robot<?>> robotUnaryOperator) {
    Robot<?> target = robotUnaryOperator.apply(null);
    ControllableVoxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    int targetW = target.getVoxels().getW();
    int targetH = target.getVoxels().getH();
    return grid -> {
      //check grid sizes
      if (grid.getW() != targetW || grid.getH() != targetH) {
        throw new IllegalArgumentException(String.format(
            "Wrong grid size: %dx%d expected, %dx%d found",
            targetW, targetH,
            grid.getW(), grid.getH()
        ));
      }
      //check grid element size
      if (grid.values().stream().anyMatch(v -> v.length != 2)) {
        Grid.Entry<double[]> firstWrong = grid.stream().filter(e -> e.getValue().length != 2).findFirst().orElse(null);
        if (firstWrong == null) {
          throw new IllegalArgumentException("Unexpected empty wrong grid item");
        }
        throw new IllegalArgumentException(String.format(
            "Wrong number of values in grid at %d,%d: %d expected, %d found",
            firstWrong.getX(), firstWrong.getY(),
            2,
            firstWrong.getValue().length
        ));
      }
      Grid<Double> strenghts = Grid.create(targetW, targetH, (x, y) -> grid.get(x, y)[0]);
      Grid<Double> phases = Grid.create(targetW, targetH, (x, y) -> grid.get(x, y)[1]);
      return previous -> {
        int previousN = previous == null ? 0 : (int) previous.getVoxels().stream().filter(Objects::nonNull).count();
        int n = Math.min(previousN + nStep, targetW * targetH);
        Grid<Double> cropped = Utils.cropGrid(gridConnected(strenghts, Double::compareTo, n), Objects::nonNull);
        Grid<ControllableVoxel> body = Grid.create(cropped, v -> (v != null) ? SerializationUtils.clone(voxelPrototype) : null);
        if (body.values().stream().noneMatch(Objects::nonNull)) {
          body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
        }
        //build controller
        TimeFunctions controller = new TimeFunctions(Grid.create(
            body.getW(),
            body.getH(),
            (x, y) -> t -> amplitude * Math.sin(2 * Math.PI * frequency * t + phases.get(x, y))
        ));
        return new Robot<>(controller, body);
      };
    };
  }

  @Override
  public Grid<double[]> exampleFor(UnaryOperator<Robot<?>> robotUnaryOperator) {
    Robot<?> target = robotUnaryOperator.apply(null);
    int targetW = target.getVoxels().getW();
    int targetH = target.getVoxels().getH();
    return Grid.create(targetW, targetH, new double[]{0d, 0d});
  }

  public static <K> Grid<K> gridConnected(Grid<K> kGrid, Comparator<K> comparator, int n) {
    Comparator<Grid.Entry<K>> entryComparator = (e1, e2) -> comparator.compare(e1.getValue(), e2.getValue());
    Predicate<Pair<Grid.Entry<?>, Grid.Entry<?>>> adjacencyPredicate = p -> (Math.abs(p.getLeft().getX() - p.getRight().getX()) <= 1 && p.getLeft().getY() == p.getRight().getY()) || (Math.abs(p.getLeft().getY() - p.getRight().getY()) <= 1 && p.getLeft().getX() == p.getRight().getX());
    Grid.Entry<K> entryFirst = kGrid.stream()
        .min(entryComparator)
        .orElseThrow(() -> new IllegalArgumentException("Grid has no max element"));
    Set<Grid.Entry<K>> selected = new HashSet<>(n);
    selected.add(entryFirst);
    while (selected.size() < n) {
      Set<Grid.Entry<K>> candidates = kGrid.stream()
          .filter(e -> e.getValue() != null)
          .filter(e -> !selected.contains(e))
          .filter(e -> selected.stream().anyMatch(f -> adjacencyPredicate.test(Pair.of(e, f))))
          .collect(Collectors.toSet());
      if (candidates.isEmpty()) {
        break;
      }
      selected.add(candidates.stream().min(entryComparator).orElse(entryFirst));
    }
    Grid<K> outGrid = Grid.create(kGrid.getW(), kGrid.getH());
    selected.forEach(e -> outGrid.set(e.getX(), e.getY(), e.getValue()));
    return outGrid;
  }

}

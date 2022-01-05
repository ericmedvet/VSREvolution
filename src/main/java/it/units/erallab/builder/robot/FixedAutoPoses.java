package it.units.erallab.builder.robot;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.behavior.PoseUtils;
import it.units.erallab.hmsrobots.core.controllers.PosesController;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author "Eric Medvet" on 2021/12/08 for VSREvolution
 */
public class FixedAutoPoses implements PrototypedFunctionBuilder<List<Integer>, Robot> {

  private final int nUniquePoses;
  private final int nPoses;
  private final int nRegions;
  private final int gridSize;
  private final double stepT;

  public FixedAutoPoses(int nUniquePoses, int nPoses, int nRegions, int gridSize, double stepT) {
    this.nUniquePoses = nUniquePoses;
    this.nPoses = nPoses;
    this.nRegions = nRegions;
    this.gridSize = gridSize;
    this.stepT = stepT;
  }

  @Override
  public Function<List<Integer>, Robot> buildFor(Robot robot) {
    Grid<Boolean> shape = Grid.create(robot.getVoxels(), Objects::nonNull);
    List<Set<Grid.Key>> availablePoses = new ArrayList<>(PoseUtils.computeClusteredByPosturePoses(
        shape,
        PoseUtils.computeClusteredByPositionPoses(shape, nRegions, 1),
        nUniquePoses, 1, new Voxel(List.of()), 4d, gridSize
    ));
    return genes -> new Robot(
        new PosesController(stepT, genes.stream()
            .map(g -> availablePoses.get(Math.max(0, Math.min(availablePoses.size() - 1, g))))
            .collect(Collectors.toList())).smoothed(4d * 2d / stepT),
        SerializationUtils.clone(robot.getVoxels())
    );
  }

  @Override
  public List<Integer> exampleFor(Robot robot) {
    return Collections.nCopies(nPoses, nUniquePoses);
  }

}

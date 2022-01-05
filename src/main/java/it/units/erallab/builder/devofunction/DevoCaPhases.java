package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.phenotype.MLP;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimeFunctions;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoCaPhases implements PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot>> {

  private final double frequency;
  private final double amplitude;
  private final MLP neuralCA;
  private final int nInitial;
  private final int nStep;
  private final double controllerStep;

  public DevoCaPhases(
      double frequency, double amplitude,
      double caInnerLayerRatio, int caNOfInnerLayers,
      int nInitial, int nStep, double controllerStep
  ) {
    this.frequency = frequency;
    this.amplitude = amplitude;
    neuralCA = new MLP(caInnerLayerRatio, caNOfInnerLayers);
    this.nInitial = nInitial;
    this.nStep = nStep;
    this.controllerStep = controllerStep;
  }

  private static class DecoratedRobot extends Robot {
    private final Grid<Double> phases;

    private DecoratedRobot(Controller controller, Grid<Voxel> voxels, Grid<Double> phases) {
      super(controller, voxels);
      this.phases = phases;
    }
  }

  @Override
  public Function<List<Double>, UnaryOperator<Robot>> buildFor(UnaryOperator<Robot> robotUnaryOperator) {
    Robot target = robotUnaryOperator.apply(null);
    Voxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
    if (voxelPrototype == null) {
      throw new IllegalArgumentException("Target robot has no valid voxels");
    }
    int neuralCaValuesSize = neuralCA.exampleFor(RealFunction.build(d -> d, 4, 2)).size();
    return list -> {
      //check values size
      if (list.size() != (neuralCaValuesSize)) {
        throw new IllegalArgumentException(String.format(
            "Wrong values size: %d expected, %d found",
            neuralCaValuesSize, list.size()
        ));
      }
      return previous -> {
        RealFunction nca = (RealFunction) neuralCA.buildFor(RealFunction.build(d -> d, 4, 2)).apply(list);
        Grid<Double> previousPhases;
        if (previous == null) {
          previousPhases = Grid.create(target.getVoxels(), v -> null);
        } else {
          if (!(previous instanceof DecoratedRobot)) {
            throw new IllegalArgumentException("Previous robot is not decorated with phases; cannot develop");
          }
          previousPhases = ((DecoratedRobot) previous).phases;
        }
        Grid<Double> phases = developBodyAndPhases(previousPhases, nca);
        Grid<Voxel> body = createBody(phases, voxelPrototype);
        //build controller
        double localAmplitude = amplitude;
        double localFrequency = frequency;
        AbstractController controller = new TimeFunctions(Grid.create(
            phases.getW(),
            phases.getH(),
            (x, y) -> t -> localAmplitude * Math.sin(2 * Math.PI * localFrequency * t + phases.get(x, y))
        ));
        if (controllerStep > 0) {
          controller = controller.step(controllerStep);
        }
        return new DecoratedRobot(
            controller,
            body,
            phases
        );
      };
    };
  }

  private Grid<Double> developBodyAndPhases(Grid<Double> previousPhases, RealFunction neuralCA) {
    int currentVoxels = (int) previousPhases.count(Objects::nonNull);
    int n = currentVoxels == 0 ? nInitial : nStep;
    for (int i = 0; i < n; i++) {
      addOnePhase(previousPhases, neuralCA);
    }
    return previousPhases;
  }


  private static void addOnePhase(Grid<Double> phases, RealFunction neuralCA) {
    int n = (int) phases.count(Objects::nonNull) + 1;
    if (n == 1) {
      phases.set(phases.getW() / 2, phases.getH() / 2, neuralCA.apply(new double[]{0d, 0d, 0d, 0d})[1]);
      return;
    }

    double defaultNegativeValue = -2d;
    Grid<Double> strengths = Grid.create(phases);
    Grid<Double> tempPhases = SerializationUtils.clone(phases);
    phases.forEach(s -> {
          if (s != null && s.value() != null) {
            strengths.set(s.key().x(), s.key().y(), defaultNegativeValue);
          } else {
            assert s != null;
            double[] neighborsPhasesValues = getNeighborsPhasesValues(phases, s.key().x(), s.key().y());
            double[] caOutputs = neuralCA.apply(neighborsPhasesValues);
            strengths.set(s.key().x(), s.key().y(), caOutputs[0]);
            tempPhases.set(s.key().x(), s.key().y(), caOutputs[1]);
          }
        }
    );

    Grid<Double> body = Utils.gridConnected(strengths, Double::compareTo, n);
    body.stream().filter(e -> e != null && e.value() != null && e.value() != defaultNegativeValue).forEach(e -> phases.set(e.key().x(), e.key().y(), tempPhases.get(e.key().x(), e.key().y()))
    );
  }

  private static double[] getNeighborsPhasesValues(Grid<Double> previous, int x, int y) {
    List<Double> neighbors = new ArrayList<>();
    IntStream.range(0, 2).map(i -> 2 * i - 1).forEach(i -> {
          Double firstNeighbor = previous.get(x + i, y);
          Double secondNeighbor = previous.get(x, y + i);
          neighbors.add(firstNeighbor == null ? 0d : firstNeighbor);
          neighbors.add(secondNeighbor == null ? 0d : secondNeighbor);
        }
    );
    return neighbors.stream().mapToDouble(d -> d).toArray();
  }

  private Grid<Voxel> createBody(Grid<Double> phases, Voxel voxelPrototype) {
    Grid<Voxel> body = Grid.create(phases, v -> v != null ? SerializationUtils.clone(voxelPrototype) : null);
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }

  @Override
  public List<Double> exampleFor(UnaryOperator<Robot> robotUnaryOperator) {
    return neuralCA.exampleFor(RealFunction.build(d -> d, 4, 2));
  }

}

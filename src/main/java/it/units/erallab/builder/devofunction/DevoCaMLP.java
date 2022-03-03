package it.units.erallab.builder.devofunction;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.builder.function.MLP;
import it.units.erallab.builder.robot.BrainHomoDistributed;
import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.util.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author "Eric Medvet" on 2021/09/29 for VSREvolution
 */
public class DevoCaMLP implements NamedProvider<PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot>>> {

  @Override
  public PrototypedFunctionBuilder<List<Double>, UnaryOperator<Robot>> build(Map<String, String> params) {
    PrototypedFunctionBuilder<List<Double>, TimedRealFunction> mlp = (new MLP()).build(
        Map.of("r", params.get("r"),
            "nIL", params.get("nIL")
        )
    );
    PrototypedFunctionBuilder<TimedRealFunction, Robot> fixedHomoDistributed = (new BrainHomoDistributed()).build(
        Map.of("s", params.get("s")
        )
    );
    PrototypedFunctionBuilder<List<Double>, TimedRealFunction> neuralCA = (new MLP()).build(
        Map.of("r", params.get("caR"),
            "nIL", params.get("caNIL")
        )
    );
    int nInitial = Integer.parseInt(params.get("s0"));
    int nStep = Integer.parseInt(params.get("nS"));
    double controllerStep = Double.parseDouble(params.getOrDefault("st","0"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, UnaryOperator<Robot>> buildFor(UnaryOperator<Robot> robotUnaryOperator) {
        Robot target = robotUnaryOperator.apply(null);
        Voxel voxelPrototype = target.getVoxels().values().stream().filter(Objects::nonNull).findFirst().orElse(null);
        if (voxelPrototype == null) {
          throw new IllegalArgumentException("Target robot has no valid voxels");
        }
        int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
        int neuralCaValuesSize = neuralCA.exampleFor(RealFunction.build(d -> d, 4, 1)).size();
        return list -> {
          //check values size
          if (list.size() != (mlpValuesSize + neuralCaValuesSize)) {
            throw new IllegalArgumentException(String.format(
                "Wrong values size: %d+%d=%d expected, %d found",
                mlpValuesSize, neuralCaValuesSize, mlpValuesSize + neuralCaValuesSize, list.size()
            ));
          }
          List<Double> listOfMLPWeights = list.subList(0, mlpValuesSize);
          List<Double> listOfCAWeights = list.subList(mlpValuesSize, list.size());

          return previous -> {
            Grid<Boolean> previousBody;
            if (previous == null) {
              previousBody = Grid.create(target.getVoxels(), v -> false);
            } else {
              previousBody = Grid.create(previous.getVoxels(), Objects::nonNull);
            }
            RealFunction realFunction = (RealFunction) neuralCA.buildFor(RealFunction.build(d -> d, 4, 1)).apply(listOfCAWeights);
            Grid<Voxel> body = createBody(developBody(previousBody, realFunction, nInitial, nStep), voxelPrototype);

            //build controller
            Robot robot = new Robot(Controller.empty(), body);
            TimedRealFunction timedRealFunction = mlp.buildFor(fixedHomoDistributed.exampleFor(target)).apply(listOfMLPWeights);
            AbstractController controller = (AbstractController) fixedHomoDistributed.buildFor(robot).apply(timedRealFunction).getController();
            if (controllerStep > 0) {
              controller = controller.step(controllerStep);
            }
            return new Robot(controller, robot.getVoxels());
          };
        };
      }

      @Override
      public List<Double> exampleFor(UnaryOperator<Robot> robotUnaryOperator) {
        Robot target = robotUnaryOperator.apply(null);
        int mlpValuesSize = mlp.exampleFor(fixedHomoDistributed.exampleFor(target)).size();
        int neuralCaValuesSize = neuralCA.exampleFor(RealFunction.build(d -> d, 4, 1)).size();
        return IntStream.range(0, mlpValuesSize + neuralCaValuesSize).mapToObj(i -> 0d).collect(Collectors.toList());
      }
    };
  }

  private Grid<Boolean> developBody(Grid<Boolean> previousBody, RealFunction neuralCA, int nInitial, int nStep) {
    int currentVoxels = (int) previousBody.count(p -> p);
    int n = currentVoxels == 0 ? nInitial : nStep;
    Grid<Boolean> nextBody = previousBody;
    for (int i = 0; i < n; i++) {
      nextBody = addOneVoxel(nextBody, neuralCA);
    }
    return nextBody;
  }

  private Grid<Voxel> createBody(Grid<Boolean> positions, Voxel voxelPrototype) {
    Grid<Voxel> body = Grid.create(positions, v -> v ? SerializationUtils.clone(voxelPrototype) : null);
    if (body.values().stream().noneMatch(Objects::nonNull)) {
      body = Grid.create(1, 1, SerializationUtils.clone(voxelPrototype));
    }
    return body;
  }

  private static Grid<Boolean> addOneVoxel(Grid<Boolean> previous, RealFunction neuralCA) {
    int n = (int) previous.count(p -> p) + 1;
    if (n == 1) {
      Grid<Boolean> newGrid = Grid.create(previous, v -> false);
      newGrid.set(newGrid.getW() / 2, newGrid.getH() / 2, true);
      return newGrid;
    }
    Grid<Double> strengths = Grid.create(previous);
    previous.forEach(s -> {
          if (s != null && s.value() != null && s.value()) {
            strengths.set(s.key().x(), s.key().y(), -2d);
          } else {
            double[] caInputs = getNeighborsValues(previous, s.key().x(), s.key().y()).stream()
                .mapToDouble(b -> b ? 1 : 0).toArray();
            strengths.set(s.key().x(), s.key().y(), neuralCA.apply(caInputs)[0]);
          }
        }
    );
    return Grid.create(Utils.gridConnected(strengths, Double::compareTo, n), Objects::nonNull);
  }

  private static List<Boolean> getNeighborsValues(Grid<Boolean> previous, int x, int y) {
    List<Boolean> neighbors = new ArrayList<>();
    IntStream.range(0, 2).map(i -> 2 * i - 1).forEach(i -> {
          neighbors.add(previous.get(x + i, y) != null && previous.get(x + i, y));
          neighbors.add(previous.get(x, y + i) != null && previous.get(x, y + i));
        }
    );
    return neighbors;
  }

}

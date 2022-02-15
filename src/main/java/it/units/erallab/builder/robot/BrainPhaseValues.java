package it.units.erallab.builder.robot;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.Voxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * @author eric
 */
public class BrainPhaseValues implements NamedProvider<PrototypedFunctionBuilder<List<Double>, Robot>> {

  @Override
  public PrototypedFunctionBuilder<List<Double>, Robot> build(Map<String, String> params) {
    double frequency = Double.parseDouble(params.getOrDefault("f", "1"));
    double amplitude = Double.parseDouble(params.getOrDefault("a","1"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Double>, Robot> buildFor(Robot robot) {
        Grid<Voxel> body = robot.getVoxels();
        long nOfVoxel = body.values().stream().filter(Objects::nonNull).count();
        return values -> {
          if (nOfVoxel != values.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values: %d expected, %d found",
                nOfVoxel,
                values.size()
            ));
          }
          int c = 0;
          Grid<Double> phases = Grid.create(body);
          for (Grid.Entry<?> entry : body) {
            if (entry.value() != null) {
              phases.set(entry.key().x(), entry.key().y(), values.get(c));
              c = c + 1;
            }
          }
          return new Robot(
              new PhaseSin(frequency, amplitude, phases),
              SerializationUtils.clone(body)
          );
        };
      }

      @Override
      public List<Double> exampleFor(Robot robot) {
        long nOfVoxel = robot.getVoxels().values().stream().filter(Objects::nonNull).count();
        return Collections.nCopies((int) nOfVoxel, 0d);
      }
    };
  }

}

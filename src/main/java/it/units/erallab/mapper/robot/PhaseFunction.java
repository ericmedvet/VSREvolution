package it.units.erallab.mapper.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.ControllableVoxel;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.mapper.PrototypedFunctionBuilder;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class PhaseFunction implements PrototypedFunctionBuilder<RealFunction, Robot<?>> {
  private final double frequency;
  private final double amplitude;

  public PhaseFunction(double frequency, double amplitude) {
    this.frequency = frequency;
    this.amplitude = amplitude;
  }

  @Override
  public Function<RealFunction, Robot<?>> buildFor(Robot<?> robot) {
    Grid<? extends ControllableVoxel> body = robot.getVoxels();
    return function -> {
      if (function.getNOfInputs() != 2) {
        throw new IllegalArgumentException(String.format(
            "Wrong number of function input args: 2 expected, %d found",
            function.getNOfInputs()
        ));
      }
      return new Robot<>(
          new PhaseSin(
              frequency,
              amplitude,
              Grid.create(
                  body.getW(),
                  body.getH(),
                  (x, y) -> function.apply(new double[]{
                      (double) x / (double) body.getW(),
                      (double) y / (double) body.getW()}
                  )[0]
              )
          ),
          SerializationUtils.clone(body)
      );
    };
  }

  @Override
  public RealFunction exampleFor(Robot<?> robot) {
    return RealFunction.from(2, 1, d -> d);
  }

}

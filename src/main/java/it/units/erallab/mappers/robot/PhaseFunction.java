package it.units.erallab.mappers.robot;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class PhaseFunction extends FixedBody<RealFunction> {
  private final double frequency;
  private final double amplitude;

  public PhaseFunction(Grid<? extends SensingVoxel> body, double frequency, double amplitude) {
    super(body);
    this.frequency = frequency;
    this.amplitude = amplitude;
  }

  @Override
  public Robot<?> apply(RealFunction function) {
    if (function.getInputDim() != 2) {
      throw new IllegalArgumentException(String.format(
          "Wrong number of function input args: 2 expected, %d found",
          function.getInputDim()
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
  }

  @Override
  public RealFunction example(Robot<?> robot) {
    return RealFunction.from(2, 1, in -> new double[]{0});
  }
}

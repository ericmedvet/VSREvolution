package it.units.erallab.mappers.robot;

import it.units.erallab.hmsrobots.core.controllers.PhaseSin;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public class PhaseValues extends FixedBody<List<Double>> {

  private final double frequency;
  private final double amplitude;

  public PhaseValues(Grid<? extends SensingVoxel> body, double frequency, double amplitude) {
    super(body);
    this.frequency = frequency;
    this.amplitude = amplitude;
  }

  @Override
  public Robot<?> apply(List<Double> values) {
    long nOfVoxel = body.values().stream().filter(Objects::nonNull).count();
    if (nOfVoxel != values.size()) {
      throw new IllegalArgumentException(String.format(
          "Wrong number of values: %d expected, %d found",
          nOfVoxel,
          values.size()
      ));
    }
    int c = 0;
    Grid<Double> phases = Grid.create(body);
    for (Grid.Entry<? extends SensingVoxel> entry : body) {
      if (entry.getValue() != null) {
        phases.set(entry.getX(), entry.getY(), values.get(c));
        c = c + 1;
      }
    }
    return new Robot<>(
        new PhaseSin(frequency, amplitude, null),
        SerializationUtils.clone(body)
    );
  }

  @Override
  public List<Double> example(Robot<?> robot) {
    return Collections.nCopies((int) body.values().stream().filter(Objects::nonNull).count(), 0d);
  }
}

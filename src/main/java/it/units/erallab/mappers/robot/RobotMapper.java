package it.units.erallab.mappers.robot;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.mappers.ReversableFunction;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public interface RobotMapper<T> extends ReversableFunction<T, Robot<?>> {
  default T example() {
    return example(null);
  }
}

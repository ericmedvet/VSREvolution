package it.units.erallab.mappers.robot;

import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public abstract class FixedBody<T> implements RobotMapper<T> {
  protected final Grid<? extends SensingVoxel> body;

  public FixedBody(Grid<? extends SensingVoxel> body) {
    this.body = body;
  }
}

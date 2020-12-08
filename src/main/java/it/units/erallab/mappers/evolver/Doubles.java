package it.units.erallab.mappers.evolver;

import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.mappers.ReversableMapper;

import java.util.List;

/**
 * @author eric
 * @created 2020/12/07
 * @project VSREvolution
 */
public abstract class Doubles<T> implements EvolverMapper<List<Double>, T> {
  protected final ReversableMapper<Grid<? extends SensingVoxel>, List<Double>, T> mapper;

  public Doubles(ReversableMapper<Grid<? extends SensingVoxel>, List<Double>, T> mapper) {
    this.mapper = mapper;
  }
}

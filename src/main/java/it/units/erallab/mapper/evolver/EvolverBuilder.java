package it.units.erallab.mapper.evolver;

import it.units.erallab.mapper.PrototypedFunctionBuilder;
import it.units.malelab.jgea.core.evolver.Evolver;

/**
 * @author eric
 */
public interface EvolverBuilder<G> {
  <T> Evolver<G, T, Double> build(PrototypedFunctionBuilder<G, T> builder, T target);
}

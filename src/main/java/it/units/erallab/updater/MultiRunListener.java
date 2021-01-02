package it.units.erallab.updater;

import it.units.malelab.jgea.core.listener.Event;
import it.units.malelab.jgea.core.listener.Listener;

import java.util.Collection;

/**
 * @author eric on 2021/01/02 for VSREvolution
 */
public interface MultiRunListener<G, S, F> extends Listener<G, S, F> {

  void listen(Collection<S> solution);

  void shutdown();

  static MultiRunListener<Object, Object, Object> deaf() {
    return new MultiRunListener<>() {
      @Override
      public void listen(Collection<Object> solution) {
      }

      @Override
      public void listen(Event<?, ?, ?> event) {
      }

      @Override
      public void shutdown() {
      }
    };
  }

}

package it.units.erallab.mappers;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public interface ReversableMapper<P, A, B> extends Function<P, Function<A, B>> {
  A example(P p);

  static <P1, A1, A> ReversableMapper<P1, A1, A1> identityOn(ReversableMapper<P1, A1, ?> nextMapper) {
    return new ReversableMapper<>() {
      @Override
      public A1 example(P1 p) {
        return nextMapper.example(p);
      }

      @Override
      public Function<A1, A1> apply(P1 p) {
        return Function.identity();
      }
    };
  }
}

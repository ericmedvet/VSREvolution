package it.units.erallab.mappers;

import java.util.function.Function;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public interface ReversableFunction<A, B> extends Function<A, B> {
  A example(B b);

  static <A1> ReversableFunction<A1, A1> identity() {
    return new ReversableFunction<>() {
      @Override
      public A1 example(A1 a) {
        return a;
      }

      @Override
      public A1 apply(A1 a) {
        return a;
      }
    };
  }
}

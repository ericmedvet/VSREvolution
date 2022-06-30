package it.units.erallab.builder.misc;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TernaryToDoubles implements NamedProvider<PrototypedFunctionBuilder<List<Integer>, List<Double>>> {

  @Override
  public PrototypedFunctionBuilder<List<Integer>, List<Double>> build(Map<String, String> params) {
    double value = Double.parseDouble(params.getOrDefault("v", "1"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<List<Integer>, List<Double>> buildFor(List<Double> doubles) {
        return integers -> {
          if (doubles.size() != integers.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values: %d expected, %d found",
                doubles.size(),
                integers.size()
            ));
          }
          return integers.stream().map(i -> (i - 1) * value).collect(Collectors.toList());
        };
      }

      @Override
      public List<Integer> exampleFor(List<Double> doubles) {
        return doubles.stream().map(d -> 3).collect(Collectors.toList());
      }
    };
  }
}

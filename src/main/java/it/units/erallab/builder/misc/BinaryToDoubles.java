package it.units.erallab.builder.misc;

import it.units.erallab.builder.NamedProvider;
import it.units.erallab.builder.PrototypedFunctionBuilder;
import it.units.malelab.jgea.representation.sequence.bit.BitString;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class BinaryToDoubles implements NamedProvider<PrototypedFunctionBuilder<BitString, List<Double>>> {

  @Override
  public PrototypedFunctionBuilder<BitString, List<Double>> build(Map<String, String> params) {
    double value = Double.parseDouble(params.getOrDefault("v", "1"));
    return new PrototypedFunctionBuilder<>() {
      @Override
      public Function<BitString, List<Double>> buildFor(List<Double> doubles) {
        return bitString -> {
          if (doubles.size() != bitString.size()) {
            throw new IllegalArgumentException(String.format(
                "Wrong number of values: %d expected, %d found",
                doubles.size(),
                bitString.size()
            ));
          }
          return bitString.stream().map(b -> b ? value : -value).collect(Collectors.toList());
        };
      }

      @Override
      public BitString exampleFor(List<Double> doubles) {
        return new BitString(doubles.size());
      }
    };
  }
}

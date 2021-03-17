package it.units.erallab.utils;

import it.units.erallab.hmsrobots.core.controllers.snn.SpikingNeuron;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYSeries;
import org.knowm.xchart.style.Styler;
import org.knowm.xchart.style.markers.SeriesMarkers;

import java.util.ArrayList;
import java.util.List;

public class MembraneEvolutionPlotter {

  public void plotMembranePotentialEvolution(SpikingNeuron spikingNeuron, int width, int height) {
    new SwingWrapper<>(getMembranePotentialEvolutionChart(spikingNeuron, width, height)).displayChart();
  }

  public void plotMembranePotentialEvolutionWithInputSpikes(SpikingNeuron spikingNeuron, int width, int height) {
    new SwingWrapper<>(getMembranePotentialEvolutionWithInputSpikesChart(spikingNeuron, width, height)).displayChart();
  }

  public XYChart getMembranePotentialEvolutionChart(SpikingNeuron spikingNeuron, int width, int height) {
    XYChart chart = new XYChart(width, height);
    chart.getStyler().setXAxisMin(0.0).setXAxisMax(spikingNeuron.getLastEvaluatedTime()).setLegendPosition(Styler.LegendPosition.InsideNW);
    if (!spikingNeuron.getMembranePotentialValues().isEmpty()) {
      XYSeries membranePotentialSeries = chart.addSeries("membrane potential", new ArrayList<>(spikingNeuron.getMembranePotentialValues().keySet()), new ArrayList<>(spikingNeuron.getMembranePotentialValues().values()));
      membranePotentialSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line).setMarker(SeriesMarkers.NONE);
    }
    XYSeries thresholdSeries = chart.addSeries("threshold", List.of(0, spikingNeuron.getLastEvaluatedTime()), List.of(spikingNeuron.getThresholdPotential(), spikingNeuron.getThresholdPotential()));
    thresholdSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Line).setMarker(SeriesMarkers.NONE);
    return chart;
  }

  public XYChart getMembranePotentialEvolutionWithInputSpikesChart(SpikingNeuron spikingNeuron, int width, int height) {
    XYChart chart = getMembranePotentialEvolutionChart(spikingNeuron, width, height);
    if (!spikingNeuron.getInputSpikesValues().isEmpty()) {
      XYSeries inputSpikesSeries = chart.addSeries("input", new ArrayList<>(spikingNeuron.getInputSpikesValues().keySet()), new ArrayList<>(spikingNeuron.getInputSpikesValues().values()));
      inputSpikesSeries.setXYSeriesRenderStyle(XYSeries.XYSeriesRenderStyle.Scatter).setMarker(SeriesMarkers.DIAMOND);
    }
    return chart;
  }

}
